"""
MULocBench to LocBench Format Converter

Converts MULocBench dataset to LocBench format for use with LocAgent evaluation.

Key output format:
- `edit_functions` field in format ["file.py:Class.func" or "file.py:func"]
- Compatible with LocAgent's evaluation.eval_metric.evaluate_results

Usage:
    python convert_mulocbench_to_locbench.py \
        --output_dir evaluation/datasets/mulocbench \
        --output_format jsonl
"""

import argparse
import ast
import json
import os
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download


REPO_ID = "somethingone/MULocBench"
DEFAULT_INPUT_FILE = "all_issues_with_pr_commit_comment_all_project_0922.json"


def load_mulocbench(input_path: Optional[str]) -> List[Dict]:
    """Load MULocBench from file or HuggingFace"""
    if input_path:
        path = input_path
    else:
        print(f"Downloading MULocBench from HuggingFace ({REPO_ID})...")
        path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=DEFAULT_INPUT_FILE, 
            repo_type="dataset"
        )
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_instance_id(issue: Dict) -> str:
    """Create instance_id in format: organization__repo_name-issue_number"""
    issue_num = str(issue.get("iss_html_url", "")).rstrip("/").split("/")[-1]
    organization = issue.get("organization", "unknown-org")
    repo_name = issue.get("repo_name", "unknown-repo")
    return f"{organization}__{repo_name}-{issue_num}"


def build_problem_statement(issue: Dict) -> str:
    """Combine title and body into problem_statement"""
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def parse_loc_key(loc_key: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """
    Parse location key like "('ClassName', 'func_name', 123)" 
    Returns (class_name, function_name) tuple
    """
    try:
        cls_name, func_name, _line = ast.literal_eval(loc_key)
    except Exception:
        return None

    cls_name = None if cls_name in (None, "None") else str(cls_name)
    func_name = None if func_name in (None, "None") else str(func_name)
    return (cls_name, func_name)


def extract_edit_functions(issue: Dict, exclude_tests: bool = True) -> List[str]:
    """
    Extract edit_functions in LocBench format.
    
    Format: ["file.py:ClassName.function_name"] or ["file.py:function_name"]
    
    Args:
        issue: MULocBench issue record
        exclude_tests: If True, skip test files (recommended for localization evaluation)
    
    Returns:
        List of location strings in LocBench format
    """
    edit_functions: List[str] = []
    
    file_loc = issue.get("file_loc")
    if not file_loc:
        return edit_functions
    
    files = file_loc.get("files", [])
    
    for file_entry in files:
        path = file_entry.get("path")
        if not path:
            continue
        
        # Optionally skip test files
        if exclude_tests and ("test" in path.lower() or path.startswith("tests/")):
            continue

        loc_map = file_entry.get("Loc") or {}
        
        for loc_key in loc_map.keys():
            if loc_key == "(None, None, None)":
                continue
                
            parsed = parse_loc_key(loc_key)
            if not parsed:
                continue

            cls_name, func_name = parsed
            
            # Skip if no function or class
            if not cls_name and not func_name:
                continue
            
            # Build the edit_functions entry in LocBench format
            if func_name and cls_name:
                # Class method: file.py:ClassName.method_name
                entity = f"{path}:{cls_name}.{func_name}"
            elif func_name:
                # Standalone function: file.py:function_name
                entity = f"{path}:{func_name}"
            else:
                # Class only (rare): file.py:ClassName
                entity = f"{path}:{cls_name}"
            
            if entity not in edit_functions:
                edit_functions.append(entity)
    
    return edit_functions


def extract_edited_files(issue: Dict, exclude_tests: bool = True) -> List[str]:
    """Extract list of edited file paths"""
    files: List[str] = []
    
    file_loc = issue.get("file_loc")
    if not file_loc:
        return files
    
    for file_entry in file_loc.get("files", []):
        path = file_entry.get("path")
        if not path:
            continue
        
        if exclude_tests and ("test" in path.lower() or path.startswith("tests/")):
            continue
            
        if path not in files:
            files.append(path)
    
    return files


def convert_issue(issue: Dict, exclude_tests: bool = True) -> Dict:
    """
    Convert MULocBench issue to LocBench format.
    
    LocBench schema (from czlll/Loc-Bench_V1):
    - repo: string
    - instance_id: string  
    - base_commit: string
    - patch: string
    - test_patch: string
    - problem_statement: string
    - hints_text: string
    - created_at: int64
    - labels: sequence
    - category: string
    - edit_functions: sequence  <-- CRITICAL for evaluation
    - added_functions: sequence
    - edit_functions_length: int64
    """
    instance_id = make_instance_id(issue)
    repo_full = f"{issue.get('organization')}/{issue.get('repo_name')}"
    edit_functions = extract_edit_functions(issue, exclude_tests=exclude_tests)
    edited_files = extract_edited_files(issue, exclude_tests=exclude_tests)
    
    return {
        "instance_id": instance_id,
        "repo": repo_full,
        "base_commit": issue.get("base_commit"),
        "problem_statement": build_problem_statement(issue),
        "patch": "",  # Empty for inference
        "test_patch": "",  # Empty for inference
        "hints_text": "",  
        "created_at": 0,  
        "labels": [],  
        "category": "",  
        "edit_functions": edit_functions,  # Ground truth for evaluation
        "added_functions": [],  
        "edit_functions_length": len(edit_functions),
        # Additional metadata (prefixed with _ to indicate non-standard)
        "_edited_files": edited_files,
    }


def print_statistics(converted: List[Dict]):
    """Print conversion statistics"""
    total_functions = sum(len(c["edit_functions"]) for c in converted)
    issues_with_functions = sum(1 for c in converted if c["edit_functions"])
    
    # Distribution of functions per issue
    func_counts = [len(c["edit_functions"]) for c in converted]
    
    print(f"\nConversion Statistics:")
    print(f"  Total issues: {len(converted)}")
    print(f"  Issues with edit_functions: {issues_with_functions} ({issues_with_functions/len(converted)*100:.1f}%)")
    print(f"  Total edit_functions: {total_functions}")
    print(f"  Avg functions per issue: {total_functions/len(converted):.2f}")
    print(f"  Max functions in single issue: {max(func_counts)}")
    print(f"  Issues with 1 function: {sum(1 for c in func_counts if c == 1)}")
    print(f"  Issues with 2-5 functions: {sum(1 for c in func_counts if 2 <= c <= 5)}")
    print(f"  Issues with 5+ functions: {sum(1 for c in func_counts if c > 5)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MULocBench to LocBench format for LocAgent evaluation"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to MULocBench JSON file. If omitted, download from HuggingFace.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/datasets/mulocbench",
        help="Directory to write converted dataset.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (jsonl recommended for LocAgent compatibility)",
    )
    parser.add_argument(
        "--include_tests",
        action="store_true",
        help="Include test files in edit_functions (default: exclude)",
    )
    args = parser.parse_args()

    print("Loading MULocBench...")
    issues = load_mulocbench(args.input_file)
    print(f"Loaded {len(issues)} issues")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert to LocBench format
    converted = []
    for issue in issues:
        converted.append(convert_issue(issue, exclude_tests=not args.include_tests))
    
    # Print statistics
    print_statistics(converted)

    # Write output
    if args.output_format == "jsonl":
        output_path = os.path.join(args.output_dir, "mulocbench.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in converted:
                f.write(json.dumps(record) + "\n")
    else:
        output_path = os.path.join(args.output_dir, "mulocbench.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2)

    print(f"\nWrote converted dataset to: {output_path}")
    
    # Print sample for verification
    print("\n=== Sample converted record ===")
    sample = next((c for c in converted if c["edit_functions"]), converted[0])
    print(f"instance_id: {sample['instance_id']}")
    print(f"repo: {sample['repo']}")
    print(f"base_commit: {sample['base_commit'][:12]}...")
    print(f"edit_functions ({len(sample['edit_functions'])} total):")
    for ef in sample['edit_functions'][:5]:
        print(f"  - {ef}")
    if len(sample['edit_functions']) > 5:
        print(f"  ... and {len(sample['edit_functions']) - 5} more")


if __name__ == "__main__":
    main()
