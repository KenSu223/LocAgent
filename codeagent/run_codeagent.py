#!/usr/bin/env python3
"""
Run CodeAgent for file localization on MULocBench issues.

This script:
1. Loads issues from MULocBench dataset
2. For each issue, checkouts the base_commit
3. Runs CodeAgent with file exploration tools
4. Saves localization results to results.json
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # codeagent directory (for models)
sys.path.insert(0, str(Path(__file__).parent.parent / "index_based"))

from models import get_smolagent_model

from smolagents import CodeAgent, tool


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_REPOS = ["pandas", "transformers", "scikit-learn", "flask", "localstack"]


# ============================================================================
# File Exploration Tools for CodeAgent
# ============================================================================

@tool
def list_directory(path: str) -> str:
    """
    List files and directories in the given path.

    Args:
        path: Directory path to list (relative to repo root)

    Returns:
        List of files and directories with their types
    """
    repo_path = os.environ.get("REPO_PATH", ".")
    full_path = os.path.join(repo_path, path) if path else repo_path

    try:
        if not os.path.exists(full_path):
            return f"Error: Path '{path}' does not exist"

        if not os.path.isdir(full_path):
            return f"Error: '{path}' is not a directory"

        entries = []
        for entry in sorted(os.listdir(full_path)):
            entry_path = os.path.join(full_path, entry)
            if os.path.isdir(entry_path):
                entries.append(f"[DIR]  {entry}/")
            else:
                size = os.path.getsize(entry_path)
                entries.append(f"[FILE] {entry} ({size} bytes)")

        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def read_file(path: str, start_line: int = 1, end_line: int = 100) -> str:
    """
    Read contents of a file from the repository.

    Args:
        path: File path relative to repo root
        start_line: Starting line number (1-indexed, default 1)
        end_line: Ending line number (default 100)

    Returns:
        File contents with line numbers
    """
    repo_path = os.environ.get("REPO_PATH", ".")
    full_path = os.path.join(repo_path, path)

    try:
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist"

        if os.path.isdir(full_path):
            return f"Error: '{path}' is a directory, not a file"

        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)

        result = [f"File: {path} (lines {start_line}-{end_idx} of {total_lines})"]
        result.append("-" * 60)

        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
            result.append(f"{i:4d} | {line.rstrip()}")

        return "\n".join(result)
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def search_in_files(pattern: str, file_pattern: str = "*.py") -> str:
    """
    Search for a pattern in files using grep.

    Args:
        pattern: Text pattern to search for
        file_pattern: File glob pattern (default: *.py)

    Returns:
        Matching lines with file paths and line numbers
    """
    repo_path = os.environ.get("REPO_PATH", ".")

    try:
        # Use grep to search
        cmd = [
            "grep", "-rn", "--include", file_pattern,
            pattern, repo_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{pattern}' in {file_pattern} files"

        # Limit output to first 50 matches
        lines = output.split("\n")
        if len(lines) > 50:
            output = "\n".join(lines[:50])
            output += f"\n... and {len(lines) - 50} more matches"

        # Make paths relative
        output = output.replace(repo_path + "/", "")

        return output
    except subprocess.TimeoutExpired:
        return "Search timed out"
    except Exception as e:
        return f"Error searching: {e}"


@tool
def get_file_structure(max_depth: int = 3) -> str:
    """
    Get the directory tree structure of the repository.

    Args:
        max_depth: Maximum depth to traverse (default 3)

    Returns:
        Tree structure of the repository
    """
    repo_path = os.environ.get("REPO_PATH", ".")

    def build_tree(path: str, prefix: str = "", depth: int = 0) -> List[str]:
        if depth >= max_depth:
            return []

        result = []
        try:
            entries = sorted(os.listdir(path))
            # Filter out hidden files and common non-essential directories
            entries = [e for e in entries if not e.startswith('.')
                      and e not in ['__pycache__', 'node_modules', '.git', 'venv', 'env']]

            for i, entry in enumerate(entries):
                entry_path = os.path.join(path, entry)
                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "

                if os.path.isdir(entry_path):
                    result.append(f"{prefix}{connector}{entry}/")
                    extension = "    " if is_last else "│   "
                    result.extend(build_tree(entry_path, prefix + extension, depth + 1))
                else:
                    result.append(f"{prefix}{connector}{entry}")
        except PermissionError:
            pass

        return result

    tree = [os.path.basename(repo_path) + "/"]
    tree.extend(build_tree(repo_path))

    return "\n".join(tree[:200])  # Limit output


# ============================================================================
# Main Logic
# ============================================================================

def checkout_commit(repo_path: str, commit: str) -> bool:
    """Checkout a specific commit in the repository."""
    try:
        # First fetch to ensure we have the commit
        subprocess.run(
            ["git", "fetch", "--unshallow"],
            cwd=repo_path, capture_output=True, timeout=60
        )
    except:
        pass

    try:
        # Reset any changes
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=repo_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path, capture_output=True, check=True
        )
        # Checkout the commit
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Warning: Could not checkout {commit[:8]}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"  Error during checkout: {e}")
        return False


def run_localization(
    agent: CodeAgent,
    issue: Dict[str, Any],
    repo_path: str
) -> Dict[str, Any]:
    """Run the agent to localize files for an issue."""

    # Build the prompt
    title = issue.get("title", "")
    body = issue.get("body", "")[:3000]  # Limit body length

    prompt = f"""You are a code localization expert. Your task is to identify which files in this repository need to be modified to fix the following issue.

## Issue Title
{title}

## Issue Description
{body}

## Instructions
1. First, explore the repository structure using get_file_structure()
2. Search for relevant code using search_in_files()
3. Read specific files using read_file() to understand the code
4. Based on your analysis, identify the files that need to be modified

## Required Output
After your analysis, provide your final answer as a JSON object with this format:
{{
    "files_to_modify": ["path/to/file1.py", "path/to/file2.py"],
    "reasoning": "Brief explanation of why these files need changes"
}}

Only include files that actually need to be modified to fix the issue.
"""

    # Set repo path in environment for tools
    os.environ["REPO_PATH"] = repo_path

    try:
        result = agent.run(prompt)
        return {
            "success": True,
            "output": str(result),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": str(e)
        }


def extract_files_from_output(output: str) -> List[str]:
    """Extract file paths from agent output."""
    import re

    files = []

    # Try to parse JSON from output
    json_match = re.search(r'\{[^{}]*"files_to_modify"[^{}]*\}', output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            files = data.get("files_to_modify", [])
        except json.JSONDecodeError:
            pass

    # Fallback: extract .py file paths
    if not files:
        py_files = re.findall(r'[\w/]+\.py', output)
        files = list(set(py_files))

    return files


def load_issues(dataset_path: str, repo_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load issues from MULocBench, optionally filtered by repository name."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    if repo_name:
        issues = [d for d in data if d.get('repo_name') == repo_name]
    else:
        issues = data
    return issues


def get_issue_result_path(output_dir: Path, issue: Dict[str, Any]) -> Path:
    """Get the path for saving an individual issue result."""
    repo_name = issue.get("repo_name", "unknown")
    issue_url = issue.get("iss_html_url", "unknown")
    # Extract issue number from URL or use hash
    if "/issues/" in issue_url:
        issue_id = issue_url.split("/issues/")[-1].split("/")[0]
    else:
        issue_id = str(hash(issue_url) % 100000)

    # Flat structure: all issues in one folder with repo_name prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{repo_name}_issue_{issue_id}.json"


def save_issue_result(output_dir: Path, issue: Dict[str, Any], result: Dict[str, Any]):
    """Save a single issue result to its own file."""
    result_path = get_issue_result_path(output_dir, issue)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result_path


def is_issue_cached(output_dir: Path, issue: Dict[str, Any]) -> bool:
    """Check if an issue result already exists and was successful."""
    result_path = get_issue_result_path(output_dir, issue)
    if not result_path.exists():
        return False
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data.get("success", False)
    except Exception:
        return False


def load_cached_results(output_dir: Path, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load all cached successful results from output directory."""
    cached = {}
    if not output_dir.exists():
        return cached

    # Scan all issue files in flat structure
    for result_file in output_dir.glob("*_issue_*.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            issue_url = result.get("issue_url")
            if issue_url and result.get("success"):
                cached[issue_url] = result
        except Exception:
            continue

    if verbose and cached:
        print(f"Loaded {len(cached)} cached successful results from {output_dir}")
    return cached


def aggregate_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Aggregate all individual result files into a list."""
    results = []
    if not output_dir.exists():
        return results

    # Scan all issue files in flat structure
    for result_file in output_dir.glob("*_issue_*.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(result)
        except Exception:
            continue
    return results


def save_aggregated_results(output_dir: Path, model_id: str, repo_filter: Optional[str] = None):
    """Save aggregated results.json with all results."""
    if not output_dir.exists():
        return

    # Aggregate all results
    all_results = aggregate_results(output_dir)
    if not all_results:
        return

    # Calculate overall metrics
    successful = sum(1 for r in all_results if r.get("success"))
    total_gt = sum(len(r.get("ground_truth_files", [])) for r in all_results)
    total_correct = sum(
        len(set(r.get("ground_truth_files", [])) & set(r.get("predicted_files", [])))
        for r in all_results
    )

    # Save overall results.json
    overall_results_path = output_dir / "results.json"
    with open(overall_results_path, 'w') as f:
        json.dump({
            "metadata": {
                "model_id": model_id,
                "repo_filter": repo_filter,
                "timestamp": datetime.now().isoformat(),
                "total_issues": len(all_results),
                "successful": successful,
                "recall": total_correct / total_gt if total_gt > 0 else 0
            },
            "results": all_results
        }, f, indent=2)


def process_repo_issues(
    repo_name: str,
    issues: List[Dict[str, Any]],
    repos_dir: Path,
    model_id: str,
    output_dir: Optional[Path] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Process all issues for a single repository sequentially.
    This function is designed to be called by a worker process.

    Args:
        repo_name: Name of the repository
        issues: List of issues for this repository
        repos_dir: Path to the repos directory
        model_id: Model ID to use for the agent
        output_dir: Directory to save individual issue results
        use_cache: Whether to skip cached issues

    Returns:
        List of result dictionaries for each issue
    """
    results = []
    repo_path = repos_dir / repo_name

    # Check repo exists
    if not repo_path.exists():
        print(f"[{repo_name}] Error: Repository not found at {repo_path}")
        for issue in issues:
            result_entry = {
                "issue_url": issue.get("iss_html_url", "unknown"),
                "repo_name": repo_name,
                "title": issue.get("title", ""),
                "base_commit": issue.get("base_commit", ""),
                "ground_truth_files": get_ground_truth_files(issue),
                "predicted_files": [],
                "success": False,
                "error": f"Repository not found: {repo_name}"
            }
            results.append(result_entry)
            if output_dir:
                save_issue_result(output_dir, issue, result_entry)
        return results

    # Filter out cached issues if using cache
    if use_cache and output_dir:
        issues_to_process = [
            issue for issue in issues
            if not is_issue_cached(output_dir, issue)
        ]
        if len(issues_to_process) < len(issues):
            print(f"[{repo_name}] Skipping {len(issues) - len(issues_to_process)} cached issues")
    else:
        issues_to_process = issues

    if not issues_to_process:
        print(f"[{repo_name}] All issues already cached, skipping repo")
        return results

    # Create agent for this worker
    print(f"[{repo_name}] Initializing CodeAgent with {model_id}...")
    model = get_smolagent_model(model_id)
    agent = CodeAgent(
        tools=[list_directory, read_file, search_in_files, get_file_structure],
        model=model,
        max_steps=10,
        verbosity_level=1
    )

    for i, issue in enumerate(issues_to_process, 1):
        issue_url = issue.get("iss_html_url", "unknown")
        title = issue.get("title", "")[:60]
        base_commit = issue.get("base_commit", "")
        gt_files = get_ground_truth_files(issue)

        # Dynamic per-issue cache check (in case another worker saved this issue)
        if use_cache and output_dir and is_issue_cached(output_dir, issue):
            print(f"\n[{repo_name}] Issue {i}/{len(issues_to_process)}: {title}... CACHED (skipping)")
            continue

        print(f"\n[{repo_name}] Issue {i}/{len(issues_to_process)}: {title}...")
        print(f"[{repo_name}] Base commit: {base_commit[:8] if base_commit else 'N/A'}")

        # Checkout base commit
        if base_commit:
            print(f"[{repo_name}] Checking out {base_commit[:8]}...")
            if not checkout_commit(str(repo_path), base_commit):
                print(f"[{repo_name}] Failed to checkout, skipping...")
                result_entry = {
                    "issue_url": issue_url,
                    "repo_name": repo_name,
                    "title": issue.get("title", ""),
                    "base_commit": base_commit,
                    "ground_truth_files": gt_files,
                    "predicted_files": [],
                    "success": False,
                    "error": "Failed to checkout commit"
                }
                results.append(result_entry)
                if output_dir:
                    save_issue_result(output_dir, issue, result_entry)
                continue

        # Run agent
        print(f"[{repo_name}] Running CodeAgent...")
        result = run_localization(agent, issue, str(repo_path))

        # Extract predicted files
        predicted_files = []
        if result["success"] and result["output"]:
            predicted_files = extract_files_from_output(result["output"])

        print(f"[{repo_name}] Predicted files: {predicted_files}")

        # Calculate metrics
        gt_set = set(gt_files)
        pred_set = set(predicted_files)
        correct = gt_set & pred_set

        if gt_set:
            recall = len(correct) / len(gt_set)
            print(f"[{repo_name}] Recall: {recall:.2%} ({len(correct)}/{len(gt_set)})")

        # Store result
        result_entry = {
            "issue_url": issue_url,
            "repo_name": repo_name,
            "title": issue.get("title", ""),
            "base_commit": base_commit,
            "ground_truth_files": gt_files,
            "predicted_files": predicted_files,
            "success": result["success"],
            "raw_output": result["output"][:2000] if result["output"] else None,
            "error": result["error"]
        }
        results.append(result_entry)

        # Save immediately after each issue (no locking needed - unique files)
        if output_dir:
            result_path = save_issue_result(output_dir, issue, result_entry)
            print(f"[{repo_name}] Saved result to {result_path}")

    print(f"\n[{repo_name}] Completed processing {len(issues_to_process)} issues")
    return results


def get_ground_truth_files(issue: Dict[str, Any]) -> List[str]:
    """Extract ground truth file paths from issue."""
    file_loc = issue.get("file_loc", {})
    files = file_loc.get("files", [])

    gt_files = []
    for f in files:
        if isinstance(f, dict):
            path = f.get("path", "")
        else:
            path = str(f)
        if path:
            gt_files.append(path)

    return gt_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CodeAgent for file localization")
    parser.add_argument("--model_id", default="gpt-4.1", help="Model to use (uses models.get_smolagent_model)")
    parser.add_argument("--repo", default=None, help="Repository name to filter (default: DEFAULT_REPOS list)")
    parser.add_argument("--max-issues", type=int, default=None, help="Max issues to process (default: all)")
    parser.add_argument("--output", default=None, help="Output file path (default: output/codeagent/results.json)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (each worker handles one repo)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached results and rerun all issues")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / "dataset" / "mulocbench.json"
    repos_dir = project_root / "dataset" / "repos"

    # Output directory for per-issue results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "output" / "codeagent" / "results"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"CodeAgent File Localization")
    print(f"=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Repository filter: {args.repo if args.repo else DEFAULT_REPOS}")
    print(f"Repos directory: {repos_dir}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)

    # Load issues
    if args.repo:
        # Single repo filter
        print(f"\nLoading issues for {args.repo}...")
        issues = load_issues(str(dataset_path), args.repo)
    else:
        # Default: filter by DEFAULT_REPOS
        print(f"\nLoading issues for default repos: {DEFAULT_REPOS}...")
        all_issues = load_issues(str(dataset_path), None)
        issues = [i for i in all_issues if i.get("repo_name") in DEFAULT_REPOS]
    print(f"Found {len(issues)} issues")

    if args.max_issues:
        issues = issues[:args.max_issues]
        print(f"Processing first {len(issues)} issues")

    # Group issues by repository
    issues_by_repo = defaultdict(list)
    for issue in issues:
        repo_name = issue.get("repo_name", "unknown")
        issues_by_repo[repo_name].append(issue)

    repos = list(issues_by_repo.keys())
    print(f"\nIssues grouped into {len(repos)} repositories")
    for repo_name in repos:
        print(f"  - {repo_name}: {len(issues_by_repo[repo_name])} issues")

    print(f"\nWorkers: {args.workers}")

    # Check cache status
    use_cache = not args.no_cache
    if use_cache:
        cached_results = load_cached_results(output_dir)
        if cached_results:
            print(f"Will skip {len(cached_results)} already completed issues")
    else:
        print("Cache disabled (--no-cache), will rerun all issues")

    # Process repositories
    if args.workers == 1 or len(repos) == 1:
        # Sequential processing (single worker)
        print(f"\nProcessing {len(repos)} repos sequentially...")
        for repo_name in repos:
            repo_issues = issues_by_repo[repo_name]
            print(f"\n{'=' * 60}")
            print(f"Processing {repo_name}: {len(repo_issues)} issues")
            print(f"{'=' * 60}")

            process_repo_issues(
                repo_name=repo_name,
                issues=repo_issues,
                repos_dir=repos_dir,
                model_id=args.model_id,
                output_dir=output_dir,
                use_cache=use_cache
            )

    else:
        # Parallel processing (multiple workers)
        num_workers = min(args.workers, len(repos))
        print(f"\nUsing {num_workers} workers for {len(repos)} repositories")

        # Prepare arguments for each worker
        worker_args = [
            (repo_name, issues_by_repo[repo_name], repos_dir, args.model_id, output_dir, use_cache)
            for repo_name in repos
        ]

        # Process repos in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_repo = {
                executor.submit(process_repo_issues, *w_args): w_args[0]
                for w_args in worker_args
            }

            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    repo_results = future.result()
                    print(f"\n[MAIN] Completed {repo_name}: {len(repo_results)} issues processed")
                except Exception as e:
                    print(f"\n[MAIN] Error processing {repo_name}: {e}")

    # Final summary - aggregate all results from individual files
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Save aggregated results at repo and overall level
    save_aggregated_results(output_dir, args.model_id, args.repo)

    results = aggregate_results(output_dir)

    successful = sum(1 for r in results if r.get("success"))
    total_gt = sum(len(r.get("ground_truth_files", [])) for r in results)
    total_correct = sum(
        len(set(r.get("ground_truth_files", [])) & set(r.get("predicted_files", [])))
        for r in results
    )

    print(f"Total results: {len(results)} issues")
    print(f"Successful: {successful}")
    if total_gt > 0:
        print(f"Overall Recall: {total_correct}/{total_gt} = {total_correct/total_gt:.2%}")

    print(f"\nResults saved to:")
    print(f"  - Per-issue: {output_dir}/<repo>_issue_<id>.json")
    print(f"  - Summary:   {output_dir}/results.json")


if __name__ == "__main__":
    main()
