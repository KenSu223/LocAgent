"""
Pre-download repositories for MULocBench instances.

This script clones and checks out repos at specific commits so that
batch_build_graph.py and batch_build_bm25.py can use them without
downloading on the fly.

Repo folder format: {repo_path}/{repo_name}__{commit[:12]}
Example: /path/to/repos/ultralytics__yolov5__77415a42e597

Usage:
    python batch_download_repos.py \
        --dataset /path/to/mulocbench.jsonl \
        --repos privategpt pytorch \
        --repo_path /path/to/repos \
        --num_processes 10
"""

import argparse
import json
import os
import os.path as osp
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Set, Dict, Tuple
import torch.multiprocessing as mp
from datasets import load_dataset
from threading import Lock


def load_benchmark_dataset(dataset_name: str, split: str):
    if osp.isfile(dataset_name):
        return load_dataset("json", data_files=dataset_name, split="train")
    return load_dataset(dataset_name, split=split)


def filter_instances_by_repos(instances: List[dict], repos_filter: List[str]) -> List[dict]:
    """
    Filter instances to only include those from specified repositories.
    """
    if not repos_filter:
        return instances
    
    # Normalize filter terms
    filter_terms = [r.lower().replace("-", "").replace("_", "") for r in repos_filter]
    
    def matches_filter(instance):
        instance_id = instance.get('instance_id', '')
        repo_name = instance_id.split('__')[0].lower().replace("-", "").replace("_", "")
        
        repo_field = instance.get('repo', '')
        if repo_field:
            repo_field_name = repo_field.split('/')[-1].lower().replace("-", "").replace("_", "")
        else:
            repo_field_name = ""
        
        return any(
            term in repo_name or repo_name in term or 
            term in repo_field_name or repo_field_name in term
            for term in filter_terms
        )
    
    filtered = [inst for inst in instances if matches_filter(inst)]
    print(f'Filtered to {len(filtered)}/{len(instances)} instances matching repos: {repos_filter}')
    
    # Log matched repos
    matched_repos = {}
    for inst in filtered:
        instance_id = inst.get('instance_id', '')
        repo_name = instance_id.split('__')[0]
        matched_repos[repo_name] = matched_repos.get(repo_name, 0) + 1
    
    for repo, count in sorted(matched_repos.items()):
        print(f'  {repo}: {count} instances')
    
    return filtered


def get_repo_path_for_instance(repo_path: str, repo: str, commit: str) -> str:
    """
    Get the path where a repo should be stored.
    Format: {repo_path}/{repo_name}__{commit[:12]}
    Example: /path/to/repos/ultralytics__yolov5__77415a42e597
    """
    safe_name = repo.replace("/", "__")
    return osp.join(repo_path, f"{safe_name}__{commit[:12]}")


def get_existing_repos(repo_path: str) -> Set[str]:
    """Get set of existing repo folder names."""
    existing = set()
    if osp.exists(repo_path):
        for item in os.listdir(repo_path):
            item_path = osp.join(repo_path, item)
            # Check if it's a git repo (has .git directory)
            if osp.isdir(item_path) and osp.exists(osp.join(item_path, '.git')):
                existing.add(item)
    return existing


def clone_repo(repo: str, commit: str, target_path: str) -> Tuple[bool, str]:
    """
    Clone a repo and checkout specific commit.
    
    Returns:
        Tuple of (success, error_message)
    """
    clone_url = f"https://github.com/{repo}.git"
    
    try:
        # Clone the repo
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, target_path],
            check=True,
            capture_output=True,
            timeout=300  # 5 minute timeout for large repos
        )
        
        # Checkout specific commit
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=target_path,
            check=True,
            capture_output=True,
            timeout=60
        )
        
        return True, ""
        
    except subprocess.TimeoutExpired:
        if osp.exists(target_path):
            shutil.rmtree(target_path)
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        if osp.exists(target_path):
            shutil.rmtree(target_path)
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return False, error_msg
    except Exception as e:
        if osp.exists(target_path):
            shutil.rmtree(target_path)
        return False, str(e)


def run_download(rank, task_queue, repo_path):
    """Worker function to download repos."""
    success_count = 0
    error_count = 0
    skip_count = 0
    
    while True:
        try:
            task = task_queue.get_nowait()
        except Exception:
            # Queue is empty
            break
        
        instance_id = task['instance_id']
        repo = task['repo']
        commit = task['base_commit']
        target_path = task['target_path']
        
        # Double-check if already exists (might have been created by another worker)
        if osp.exists(target_path) and osp.exists(osp.join(target_path, '.git')):
            print(f'[{rank}] {instance_id} already exists, skipping.')
            skip_count += 1
            continue
        
        print(f'[{rank}] Downloading {instance_id}...')
        print(f'[{rank}]   repo: {repo}')
        print(f'[{rank}]   commit: {commit[:12]}')
        print(f'[{rank}]   target: {osp.basename(target_path)}')
        
        success, error_msg = clone_repo(repo, commit, target_path)
        
        if success:
            print(f'[{rank}] ✓ Downloaded {instance_id}')
            success_count += 1
        else:
            print(f'[{rank}] ✗ Error downloading {instance_id}: {error_msg}')
            error_count += 1
    
    print(f'[{rank}] Finished: {success_count} success, {error_count} errors, {skip_count} skipped')


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download repositories for MULocBench instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Repo folder format: {repo_path}/{repo_name}__{commit[:12]}
Example: /path/to/repos/ultralytics__yolov5__77415a42e597

Examples:
    # Download repos for specific repos
    python batch_download_repos.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --repos privategpt pytorch \\
        --repo_path /path/to/repos \\
        --num_processes 5

    # Download all repos in dataset
    python batch_download_repos.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --repo_path /path/to/repos \\
        --num_processes 5
        
    # Resume downloading (automatically skips existing)
    python batch_download_repos.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --repos privategpt pytorch \\
        --repo_path /path/to/repos

After downloading, you can use the repos with evaluate_cursor_cli_local.py:
    python evaluate_cursor_cli_local.py \\
        --dataset_path /path/to/mulocbench.jsonl \\
        --output_dir results/cursor_cli \\
        --repos privategpt pytorch \\
        --repo_cache_dir /path/to/repos
"""
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to MULocBench JSONL file")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repo_path", type=str, required=True,
                        help="Directory to download repos to")
    parser.add_argument("--repos", type=str, nargs="+", default=None,
                        help="Filter to specific repos (e.g., --repos privategpt pytorch)")
    parser.add_argument("--num_processes", type=int, default=2,
                        help="Number of parallel download workers (default: 5)")
    parser.add_argument("--instance_id_path", type=str, default="",
                        help="Path to a file containing a list of selected instance IDs")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    bench_data = load_benchmark_dataset(args.dataset, args.split)
    all_instances = [instance for instance in bench_data]
    print(f"Loaded {len(all_instances)} instances")
    
    # Filter by repos if specified
    if args.repos:
        all_instances = filter_instances_by_repos(all_instances, args.repos)
    
    # Filter by instance_id_path if specified
    if args.instance_id_path and osp.exists(args.instance_id_path):
        with open(args.instance_id_path, 'r') as f:
            selected_ids = set(json.loads(f.read()))
        all_instances = [inst for inst in all_instances if inst['instance_id'] in selected_ids]
        print(f"Filtered to {len(all_instances)} instances from instance_id_path")
    
    if len(all_instances) == 0:
        print("No instances to download!")
        return
    
    # Create repo directory
    os.makedirs(args.repo_path, exist_ok=True)
    
    # Check existing repos
    existing_repos = get_existing_repos(args.repo_path)
    print(f"\nFound {len(existing_repos)} existing repos in {args.repo_path}")
    
    # Build task list - each unique repo+commit combination
    # Multiple instances might share the same repo+commit
    seen_repo_commits = set()
    tasks = []
    already_done = []
    
    for instance in all_instances:
        instance_id = instance['instance_id']
        repo = instance['repo']
        commit = instance['base_commit']
        
        # Get target path
        target_path = get_repo_path_for_instance(args.repo_path, repo, commit)
        folder_name = osp.basename(target_path)
        
        # Skip if we've already seen this repo+commit
        repo_commit_key = f"{repo}__{commit[:12]}"
        if repo_commit_key in seen_repo_commits:
            continue
        seen_repo_commits.add(repo_commit_key)
        
        # Check if already exists
        if folder_name in existing_repos:
            already_done.append(instance_id)
        else:
            tasks.append({
                'instance_id': instance_id,
                'repo': repo,
                'base_commit': commit,
                'target_path': target_path
            })
    
    print(f"\nUnique repo+commit combinations: {len(seen_repo_commits)}")
    print(f"  Already downloaded (skipping): {len(already_done)}")
    print(f"  Need downloading: {len(tasks)}")
    
    if len(tasks) == 0:
        print("\nAll repos already downloaded!")
        return
    
    # Create queue
    manager = mp.Manager()
    queue = manager.Queue()
    for task in tasks:
        queue.put(task)
    
    # Start workers
    start_time = time.time()
    num_workers = min(len(tasks), args.num_processes)
    print(f"\nStarting {num_workers} workers to download {len(tasks)} repos...")
    
    mp.spawn(
        run_download,
        nprocs=num_workers,
        args=(queue, args.repo_path),
        join=True
    )
    
    end_time = time.time()
    print(f'\nTotal execution time: {end_time - start_time:.1f}s')
    
    # Final summary
    final_existing = get_existing_repos(args.repo_path)
    newly_downloaded = len(final_existing) - len(existing_repos)
    print(f"Newly downloaded repos: {newly_downloaded}")
    print(f"Total repos now: {len(final_existing)}")
    
    # Show example folder names
    print(f"\nRepo folder format: {{repo_name}}__{{commit[:12]}}")
    print(f"Example folders in {args.repo_path}:")
    for folder in sorted(final_existing)[:5]:
        print(f"  {folder}/")
    if len(final_existing) > 5:
        print(f"  ... and {len(final_existing) - 5} more")


if __name__ == "__main__":
    main()