"""
Example Usage:

# Build BM25 indexes for privategpt and pytorch only
python batch_build_bm25.py \
    --dataset /Users/kensu/Downloads/loc_agent/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --repos privategpt pytorch \
    --download_repo \
    --num_processes 10

# Resume/retry (automatically skips existing)
python batch_build_bm25.py \
    --dataset /Users/kensu/Downloads/loc_agent/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --repos privategpt pytorch \
    --download_repo

# Add a new repo to existing indexes
python batch_build_bm25.py \
    --dataset /Users/kensu/Downloads/loc_agent/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --repos numpy \
    --download_repo

"""


import argparse
import json
import os
import pickle
import time
from pathlib import Path
import subprocess
import torch.multiprocessing as mp
import os.path as osp
from datasets import load_dataset
from typing import List, Optional, Set
from util.benchmark.setup_repo import setup_repo
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo as build_code_retriever
)


def list_folders(path):
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


def load_benchmark_dataset(dataset_name: str, split: str):
    if osp.isfile(dataset_name):
        return load_dataset("json", data_files=dataset_name, split="train")
    return load_dataset(dataset_name, split=split)


def filter_instances_by_repos(instances: List[dict], repos_filter: List[str]) -> List[dict]:
    """
    Filter instances to only include those from specified repositories.
    
    Args:
        instances: List of instance dicts
        repos_filter: List of repo names to include (e.g., ['scikit-learn', 'flask'])
    
    Returns:
        Filtered list of instances
    """
    if not repos_filter:
        return instances
    
    # Normalize filter terms
    filter_terms = [r.lower().replace("-", "").replace("_", "") for r in repos_filter]
    
    def matches_filter(instance):
        # Extract repo name from instance_id (e.g., "scikit-learn__scikit-learn-12345" -> "scikit-learn")
        instance_id = instance.get('instance_id', '')
        repo_name = instance_id.split('__')[0].lower().replace("-", "").replace("_", "")
        
        # Also check the 'repo' field if available
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


def get_existing_indexes(index_dir: str) -> Set[str]:
    """Get set of instance_ids that already have BM25 indexes built."""
    existing = set()
    if osp.exists(index_dir):
        for item in os.listdir(index_dir):
            item_path = osp.join(index_dir, item)
            # BM25 indexes are directories containing corpus.jsonl
            if osp.isdir(item_path):
                corpus_file = osp.join(item_path, 'corpus.jsonl')
                if osp.exists(corpus_file):
                    existing.add(item)
    return existing


def get_repo_folder_name(instance_data: dict) -> str:
    """
    Get the folder name for a pre-downloaded repo.
    Format: {repo_name}__{commit[:12]}
    Example: pytorch__pytorch__a63524684d02
    """
    repo = instance_data.get('repo', '')
    commit = instance_data.get('base_commit', '')
    safe_name = repo.replace("/", "__")
    return f"{safe_name}__{commit[:12]}"


def run(rank, repo_queue, repo_path, out_path,
        download_repo=False, instance_data=None, similarity_top_k=10):
    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            # Queue is empty
            break

        output_file = osp.join(out_path, repo_name)
        if osp.exists(output_file):
            print(f'[{rank}] {repo_name} already processed, skipping.')
            continue

        if download_repo:
            # get process specific base dir
            repo_base_dir = str(osp.join(repo_path, str(rank)))
            os.makedirs(repo_base_dir, exist_ok=True)
            # clone and check actual repo
            try:
                repo_dir = setup_repo(instance_data=instance_data[repo_name], 
                                      repo_base_dir=repo_base_dir, 
                                      dataset=None)
            except subprocess.CalledProcessError as e:
                print(f'[{rank}] Error checkout commit {repo_name}: {e}')
                continue
        else:
            # Use repo__commit folder format for pre-downloaded repos
            if instance_data and repo_name in instance_data:
                folder_name = get_repo_folder_name(instance_data[repo_name])
                repo_dir = osp.join(repo_path, folder_name)
            else:
                # Fallback to instance_id as folder name
                repo_dir = osp.join(repo_path, repo_name)
            
            if not osp.exists(repo_dir):
                print(f'[{rank}] Repo folder not found: {repo_dir}, skipping.')
                continue

        print(f'[{rank}] Start process {repo_name}')
        try:
            retriever = build_code_retriever(repo_dir, persist_path=output_file,
                                         similarity_top_k=similarity_top_k)
            print(f'[{rank}] Processed {repo_name}')
        except Exception as e:
            print(f'[{rank}] Error processing {repo_name}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build BM25 indexes for code localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build indexes for all instances in dataset
    python batch_build_bm25.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --download_repo \\
        --num_processes 10

    # Build indexes for specific repos only
    python batch_build_bm25.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --repos scikit-learn flask pytorch \\
        --download_repo \\
        --num_processes 10
        
    # Resume building (automatically skips existing indexes)
    python batch_build_bm25.py \\
        --dataset /path/to/mulocbench.jsonl \\
        --repos privategpt pytorch \\
        --download_repo
"""
    )
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--num_processes', type=int, default=30)
    parser.add_argument('--download_repo', action='store_true', 
                        help='Whether to download the codebase to `repo_path` before indexing.')
    parser.add_argument('--repo_path', type=str, default='playground/build_graph', 
                        help='The directory where you plan to pull or have already pulled the codebase.')
    parser.add_argument('--index_dir', type=str, default='index_data', 
                        help='The base directory where the generated BM25 index will be saved.')
    parser.add_argument('--instance_id_path', type=str, default='', 
                        help='Path to a file containing a list of selected instance IDs.')
    parser.add_argument('--similarity_top_k', type=int, default=10,
                        help='Number of top similar documents to retrieve (default: 10)')
    # NEW: Add repos filter argument
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific repos (e.g., --repos scikit-learn flask pytorch privategpt)"
    )
    args = parser.parse_args()

    
    dataset_name = osp.splitext(osp.basename(args.dataset))[0]
    args.index_dir = f'{args.index_dir}/{dataset_name}/BM25_index/'
    os.makedirs(args.index_dir, exist_ok=True)
    
    # Get existing indexes for logging
    existing_indexes = get_existing_indexes(args.index_dir)
    print(f"Found {len(existing_indexes)} existing BM25 indexes in {args.index_dir}")
    
    # Always load instance data for repo folder mapping
    bench_data = load_benchmark_dataset(args.dataset, args.split)
    all_instances = [instance for instance in bench_data]
    
    # NEW: Filter by repos if specified
    if args.repos:
        all_instances = filter_instances_by_repos(all_instances, args.repos)
    
    if args.instance_id_path and osp.exists(args.instance_id_path):
        with open(args.instance_id_path, 'r') as f:
            selected_ids = set(json.loads(f.read()))
        all_instances = [inst for inst in all_instances if inst['instance_id'] in selected_ids]
        print(f"Filtered to {len(all_instances)} instances from instance_id_path")
    
    # Build instance data dict and repo folders list
    selected_instance_data = {}
    repo_folders = []
    for instance in all_instances:
        instance_id = instance['instance_id']
        repo_folders.append(instance_id)
        selected_instance_data[instance_id] = instance

    # Determine which instances need processing
    to_process = []
    already_done = []
    for repo in repo_folders:
        if repo in existing_indexes:
            already_done.append(repo)
        else:
            to_process.append(repo)
    
    print(f"\nInstances to process:")
    print(f"  Already indexed (skipping): {len(already_done)}")
    print(f"  Need indexing: {len(to_process)}")
    print(f"  Total: {len(repo_folders)}")
    
    if len(to_process) == 0:
        print("\nAll instances already indexed!")
        exit(0)

    os.makedirs(args.repo_path, exist_ok=True)

    # Create a shared queue and add only unprocessed repositories to it
    manager = mp.Manager()
    queue = manager.Queue()
    for repo in to_process:
        queue.put(repo)

    start_time = time.time()

    # Start multiprocessing with a global queue
    num_workers = min(len(to_process), args.num_processes)
    print(f"\nStarting {num_workers} workers to process {len(to_process)} instances...")
    
    mp.spawn(
        run,
        nprocs=num_workers,
        args=(queue, args.repo_path, args.index_dir,
              args.download_repo, selected_instance_data, args.similarity_top_k),
        join=True
    )

    end_time = time.time()
    print(f'\nTotal Execution time = {end_time - start_time:.3f}s')
    
    # Final summary
    final_existing = get_existing_indexes(args.index_dir)
    newly_created = len(final_existing) - len(existing_indexes)
    print(f"Newly created indexes: {newly_created}")
    print(f"Total indexes now: {len(final_existing)}")