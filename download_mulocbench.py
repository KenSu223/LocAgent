#!/usr/bin/env python3
"""
Download MULocBench dataset from Hugging Face and save as JSON.

MULocBench: https://huggingface.co/datasets/somethingone/MULocBench
Contains 1,100 issues from 46 popular GitHub Python projects.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


def download_mulocbench(output_dir: str = "../dataset") -> str:
    """
    Download MULocBench dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Path to the saved JSON file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    print("Downloading MULocBench dataset from Hugging Face...")

    # Load dataset from Hugging Face
    dataset = load_dataset("somethingone/MULocBench")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts and save as JSON
    data = []
    for split_name in dataset.keys():
        print(f"Processing split: {split_name}")
        for item in dataset[split_name]:
            data.append(dict(item))

    output_file = output_path / "mulocbench.json"

    print(f"Saving {len(data)} issues to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {output_file}")
    return str(output_file)


def extract_repos(json_file: str) -> List[Tuple[str, str]]:
    """
    Extract unique repository URLs and base commits from MULocBench.

    Args:
        json_file: Path to mulocbench.json

    Returns:
        List of (repo_url, base_commit) tuples
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract unique repos with their commits
    repos: Dict[str, Set[str]] = {}

    for item in data:
        org = item.get('organization', '')
        repo = item.get('repo_name', '')
        base_commit = item.get('base_commit', '')

        if org and repo and base_commit:
            repo_url = f"https://github.com/{org}/{repo}"
            if repo_url not in repos:
                repos[repo_url] = set()
            repos[repo_url].add(base_commit)

    # Return list of (url, first_commit) - use first commit found for each repo
    result = []
    for url, commits in sorted(repos.items()):
        # Use the first commit alphabetically for consistency
        first_commit = sorted(commits)[0]
        result.append((url, first_commit))

    return result


def generate_repos_txt(repos: List[Tuple[str, str]], output_file: str = "repos.txt"):
    """
    Generate repos.txt file with repository URLs and commits.

    Args:
        repos: List of (repo_url, commit) tuples
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for url, commit in repos:
            f.write(f"{url} {commit}\n")

    print(f"Generated {output_file} with {len(repos)} repositories")


def print_repo_stats(json_file: str):
    """Print statistics about repositories in the dataset."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repos = {}
    for item in data:
        org = item.get('organization', '')
        repo = item.get('repo_name', '')
        if org and repo:
            key = f"{org}/{repo}"
            repos[key] = repos.get(key, 0) + 1

    print(f"\nDataset Statistics:")
    print(f"  Total issues: {len(data)}")
    print(f"  Unique repositories: {len(repos)}")
    print(f"\nIssues per repository:")
    for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and process MULocBench dataset")
    parser.add_argument("--output-dir", default="../dataset", help="Output directory for dataset")
    parser.add_argument("--repos-txt", default="repos.txt", help="Output repos.txt file")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats (requires existing dataset)")
    parser.add_argument("--extract-repos", action="store_true", help="Extract repos from existing dataset")

    args = parser.parse_args()

    json_file = os.path.join(args.output_dir, "mulocbench.json")

    if args.stats_only:
        if os.path.exists(json_file):
            print_repo_stats(json_file)
        else:
            print(f"Error: {json_file} not found. Run without --stats-only first.")
        return

    if args.extract_repos:
        if os.path.exists(json_file):
            repos = extract_repos(json_file)
            generate_repos_txt(repos, args.repos_txt)
            print_repo_stats(json_file)
        else:
            print(f"Error: {json_file} not found. Run without --extract-repos first.")
        return

    # Download dataset
    json_file = download_mulocbench(args.output_dir)

    # Extract repos and generate repos.txt
    repos = extract_repos(json_file)
    generate_repos_txt(repos, args.repos_txt)

    # Print stats
    print_repo_stats(json_file)


if __name__ == "__main__":
    main()
