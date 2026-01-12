#!/usr/bin/env python3
"""
Build Knowledge Graph for a specific repository using locagent's graph builder.

Usage:
    python build_kg.py --repo <repo_name>
    python build_kg.py --repo scikit-learn
    python build_kg.py --all  # Build KGs for all default repos

The KG will be saved to /dataset/kgs/{repo_name}.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dependency_graph.build_graph import build_graph, VERSION


# Default repos (aligned with codeagent)
DEFAULT_REPOS = ["fastapi", "transformers", "scikit-learn", "flask", "localstack"]


def build_kg_for_repo(repo_name: str, repos_dir: Path, output_dir: Path, force: bool = False) -> bool:
    """
    Build knowledge graph for a single repository.

    Args:
        repo_name: Name of the repository
        repos_dir: Directory containing the repositories
        output_dir: Directory to save the KG
        force: If True, rebuild even if KG exists

    Returns:
        True if successful, False otherwise
    """
    repo_path = repos_dir / repo_name
    output_file = output_dir / f"{repo_name}.pkl"

    # Check if repo exists
    if not repo_path.exists():
        print(f"[ERROR] Repository not found: {repo_path}")
        return False

    # Check if KG already exists
    if output_file.exists() and not force:
        print(f"[SKIP] KG already exists: {output_file}")
        return True

    print(f"[BUILD] Building KG for {repo_name}...")
    print(f"  Repo path: {repo_path}")
    print(f"  Output: {output_file}")

    try:
        # Build the graph using locagent's build_graph function
        # global_import=True enables cross-module fuzzy matching
        G = build_graph(str(repo_path), global_import=True)

        # Print graph stats
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")

        # Save to pickle (aligned with locagent)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(G, f)

        print(f"[DONE] Saved KG to {output_file}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to build KG for {repo_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph for repositories")
    parser.add_argument("--repo", type=str, default=None,
                        help="Repository name to build KG for")
    parser.add_argument("--all", action="store_true",
                        help="Build KGs for all default repos")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if KG exists")
    parser.add_argument("--repos-dir", type=str, default=None,
                        help="Directory containing repositories (default: dataset/repos)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save KGs (default: dataset/kgs)")
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.repos_dir:
        repos_dir = Path(args.repos_dir)
    else:
        repos_dir = project_root / "dataset" / "repos"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "dataset" / "kgs"

    print(f"=" * 60)
    print(f"Knowledge Graph Builder (locagent {VERSION})")
    print(f"=" * 60)
    print(f"Repos directory: {repos_dir}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)

    # Determine which repos to build
    if args.all:
        repos = DEFAULT_REPOS
        print(f"\nBuilding KGs for all default repos: {repos}")
    elif args.repo:
        repos = [args.repo]
        print(f"\nBuilding KG for: {args.repo}")
    else:
        parser.print_help()
        print("\nError: Please specify --repo <name> or --all")
        sys.exit(1)

    # Build KGs
    success_count = 0
    fail_count = 0

    for repo_name in repos:
        print(f"\n{'─' * 40}")
        if build_kg_for_repo(repo_name, repos_dir, output_dir, args.force):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Success: {success_count}/{len(repos)}")
    if fail_count > 0:
        print(f"Failed: {fail_count}/{len(repos)}")
    print(f"KGs saved to: {output_dir}")


if __name__ == "__main__":
    main()
