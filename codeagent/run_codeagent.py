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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "index_based"))

from models import get_smolagent_model

from smolagents import CodeAgent, tool


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


def load_issues(dataset_path: str, repo_name: str = "manim") -> List[Dict[str, Any]]:
    """Load issues for a specific repository from MULocBench."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    issues = [d for d in data if d.get('repo_name') == repo_name]
    return issues


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
    parser.add_argument("--model", default="gpt-4.1", help="Model to use")
    parser.add_argument("--repo", default="manim", help="Repository name")
    parser.add_argument("--max-issues", type=int, default=5, help="Max issues to process")
    parser.add_argument("--output", default="results.json", help="Output file")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "dataset" / "mulocbench.json"
    repo_path = script_dir.parent / "dataset" / "repos" / args.repo
    output_path = script_dir / args.output

    print(f"=" * 60)
    print(f"CodeAgent File Localization")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Repository: {repo_path}")
    print(f"Output: {output_path}")
    print(f"=" * 60)

    # Check repo exists
    if not repo_path.exists():
        print(f"Error: Repository not found at {repo_path}")
        print(f"Please clone it first: git clone https://github.com/3b1b/manim {repo_path}")
        sys.exit(1)

    # Load issues
    print(f"\nLoading issues for {args.repo}...")
    issues = load_issues(str(dataset_path), args.repo)
    print(f"Found {len(issues)} issues")

    if args.max_issues:
        issues = issues[:args.max_issues]
        print(f"Processing first {len(issues)} issues")

    # Create agent
    print(f"\nInitializing CodeAgent with {args.model}...")
    model = get_smolagent_model(args.model)
    agent = CodeAgent(
        tools=[list_directory, read_file, search_in_files, get_file_structure],
        model=model,
        max_steps=10,
        verbosity_level=1
    )

    # Process issues
    results = []

    for i, issue in enumerate(issues, 1):
        issue_url = issue.get("iss_html_url", "unknown")
        title = issue.get("title", "")[:60]
        base_commit = issue.get("base_commit", "")
        gt_files = get_ground_truth_files(issue)

        print(f"\n{'=' * 60}")
        print(f"Issue {i}/{len(issues)}: {title}...")
        print(f"URL: {issue_url}")
        print(f"Base commit: {base_commit[:8] if base_commit else 'N/A'}")
        print(f"Ground truth files: {gt_files}")
        print(f"{'=' * 60}")

        # Checkout base commit
        if base_commit:
            print(f"Checking out {base_commit[:8]}...")
            if not checkout_commit(str(repo_path), base_commit):
                print("  Failed to checkout, skipping...")
                results.append({
                    "issue_url": issue_url,
                    "title": issue.get("title", ""),
                    "base_commit": base_commit,
                    "ground_truth_files": gt_files,
                    "predicted_files": [],
                    "success": False,
                    "error": "Failed to checkout commit"
                })
                continue

        # Run agent
        print("Running CodeAgent...")
        result = run_localization(agent, issue, str(repo_path))

        # Extract predicted files
        predicted_files = []
        if result["success"] and result["output"]:
            predicted_files = extract_files_from_output(result["output"])

        print(f"Predicted files: {predicted_files}")

        # Calculate metrics
        gt_set = set(gt_files)
        pred_set = set(predicted_files)
        correct = gt_set & pred_set

        if gt_set:
            recall = len(correct) / len(gt_set)
            print(f"Recall: {recall:.2%} ({len(correct)}/{len(gt_set)})")

        # Store result
        results.append({
            "issue_url": issue_url,
            "title": issue.get("title", ""),
            "base_commit": base_commit,
            "ground_truth_files": gt_files,
            "predicted_files": predicted_files,
            "success": result["success"],
            "raw_output": result["output"][:2000] if result["output"] else None,
            "error": result["error"]
        })

        # Save intermediate results
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "model": args.model,
                    "repo": args.repo,
                    "timestamp": datetime.now().isoformat(),
                    "total_issues": len(issues),
                    "processed": i
                },
                "results": results
            }, f, indent=2)

    # Final summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    successful = sum(1 for r in results if r["success"])
    total_gt = sum(len(r["ground_truth_files"]) for r in results)
    total_correct = sum(
        len(set(r["ground_truth_files"]) & set(r["predicted_files"]))
        for r in results
    )

    print(f"Processed: {len(results)} issues")
    print(f"Successful: {successful}")
    if total_gt > 0:
        print(f"Overall Recall: {total_correct}/{total_gt} = {total_correct/total_gt:.2%}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
