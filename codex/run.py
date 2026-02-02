#!/usr/bin/env python3
"""Run the Codex agent for file localization on MULocBench."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "mulocbench.json"
REPOS_DIR = PROJECT_ROOT / "dataset" / "repos"
OUTPUT_DIR = PROJECT_ROOT / "output" / "codex"
RESULTS_DIR = OUTPUT_DIR / "results"
CODEX_BIN = Path("/home/ubuntu/codex/codex-rs/target/release/codex-exec")

AZURE_BASE_URL = "https://cielara-research-resource.cognitiveservices.azure.com/openai/v1/"

# Import sibling modules
from codex.prompt import SYSTEM_PROMPT, build_prompt
from codex.parse_output import parse_codex_output


def load_dataset(repo_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load MULocBench and filter to locally available repos."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    available_repos = {d.name for d in REPOS_DIR.iterdir() if d.is_dir()}

    filtered = []
    for entry in data:
        repo = entry["repo_name"]
        if repo not in available_repos:
            continue
        if repo_filter and repo != repo_filter:
            continue
        filtered.append(entry)

    return filtered


def get_ground_truth_files(entry: Dict[str, Any]) -> List[str]:
    """Extract ground truth file paths from a dataset entry."""
    file_loc = entry.get("file_loc", {})
    files = file_loc.get("files", [])
    return [f["path"] for f in files if "path" in f]


def get_issue_id(entry: Dict[str, Any]) -> str:
    """Derive a short identifier for the issue (e.g. flask_issue_2264)."""
    url = entry.get("iss_html_url", "")
    # URL like https://github.com/pallets/flask/issues/2264
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        issue_num = parts[-1]
    else:
        issue_num = entry["base_commit"][:8]
    repo = entry["repo_name"]
    return f"{repo}_issue_{issue_num}"


def checkout_repo(repo_name: str, base_commit: str) -> Path:
    """Reset repo to the specified commit."""
    repo_dir = REPOS_DIR / repo_name
    subprocess.run(
        ["git", "checkout", "--force", base_commit],
        cwd=repo_dir, capture_output=True, timeout=60,
    )
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_dir, capture_output=True, timeout=60,
    )
    subprocess.run(
        ["git", "reset", "--hard", base_commit],
        cwd=repo_dir, capture_output=True, timeout=60,
    )
    return repo_dir


def get_api_key() -> str:
    """Get the Azure OpenAI API key from environment or .env file."""
    key = os.environ.get("AZURE_OPENAI_API_KEY_MAIN", "")
    if key:
        return key
    # Try loading from .env
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("AZURE_OPENAI_API_KEY_MAIN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    return ""


def run_codex(repo_dir: Path, prompt: str, api_key: str, model: str = "gpt-5.2") -> str:
    """Invoke codex exec and return raw JSONL output."""
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = AZURE_BASE_URL
    env["OPENAI_API_KEY"] = api_key

    cmd = [
        str(CODEX_BIN),
        "exec",
        "--json",
        "-m", model,
        "-C", str(repo_dir),
        "--sandbox", "read-only",
    ]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    # Combine stdout (JSONL events) and note any stderr
    output = result.stdout or ""
    if result.returncode != 0 and not output:
        raise RuntimeError(
            f"codex exec failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return output


def run_instance(entry: Dict[str, Any], api_key: str, model: str) -> Dict[str, Any]:
    """Process a single benchmark instance."""
    issue_id = get_issue_id(entry)
    repo_name = entry["repo_name"]
    base_commit = entry["base_commit"]
    title = entry["title"]
    body = entry.get("body", "")
    ground_truth = get_ground_truth_files(entry)

    result = {
        "issue_url": entry.get("iss_html_url", ""),
        "repo_name": repo_name,
        "title": title,
        "base_commit": base_commit,
        "ground_truth_files": ground_truth,
        "predicted_files": [],
        "success": False,
        "raw_output": "",
        "error": None,
    }

    try:
        # Checkout repo to the right commit
        repo_dir = checkout_repo(repo_name, base_commit)

        # Build the prompt
        user_prompt = (
            SYSTEM_PROMPT + "\n\n" + build_prompt(title, body)
        )

        # Run codex
        raw_output = run_codex(repo_dir, user_prompt, api_key, model)
        result["raw_output"] = raw_output

        # Parse output
        predicted = parse_codex_output(raw_output)
        result["predicted_files"] = predicted[:5]
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return issue_id, result


def compute_metrics(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics over all results."""
    total = 0
    acc1 = 0
    acc5 = 0
    recall1 = 0.0
    recall5 = 0.0
    successful = 0
    failed = 0

    for issue_id, r in results.items():
        if not r.get("success"):
            failed += 1
            continue

        gt = r.get("ground_truth_files", [])
        if not gt:
            continue

        total += 1
        successful += 1
        pred = r.get("predicted_files", [])

        gt_set = set(gt)
        # acc@k: all ground truth in top-k
        if gt_set.issubset(set(pred[:1])):
            acc1 += 1
        if gt_set.issubset(set(pred[:5])):
            acc5 += 1
        # recall@k
        recall1 += len(gt_set.intersection(set(pred[:1]))) / len(gt_set)
        recall5 += len(gt_set.intersection(set(pred[:5]))) / len(gt_set)

    if total == 0:
        return {"error": "No valid samples"}

    return {
        "total_samples": total,
        "successful_runs": successful,
        "failed_runs": failed,
        "metrics": {
            "acc@1": round(acc1 / total, 4),
            "acc@5": round(acc5 / total, 4),
            "recall@1": round(recall1 / total, 4),
            "recall@5": round(recall5 / total, 4),
        },
        "raw_counts": {
            "acc@1_correct": acc1,
            "acc@5_correct": acc5,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run Codex agent for file localization")
    parser.add_argument("--limit", type=int, default=0, help="Max instances to run (0=all)")
    parser.add_argument("--repo", type=str, default=None, help="Filter to a specific repo")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Model name to use")
    args = parser.parse_args()

    # Validate prerequisites
    if not CODEX_BIN.exists():
        print(f"ERROR: codex binary not found at {CODEX_BIN}")
        print("Build it with: cd /home/ubuntu/codex/codex-rs && cargo build --release")
        sys.exit(1)

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No API key found. Set AZURE_OPENAI_API_KEY_MAIN env var or add it to ~/.env")
        sys.exit(1)

    # Load dataset
    instances = load_dataset(repo_filter=args.repo)
    if args.limit > 0:
        instances = instances[:args.limit]

    print(f"Running {len(instances)} instances (model={args.model})")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for i, entry in enumerate(instances):
        issue_id = get_issue_id(entry)
        result_file = RESULTS_DIR / f"{issue_id}.json"

        # Skip if already completed
        if result_file.exists():
            try:
                existing = json.loads(result_file.read_text())
                if existing.get("success"):
                    print(f"[{i+1}/{len(instances)}] SKIP {issue_id} (already done)")
                    all_results[issue_id] = existing
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        print(f"[{i+1}/{len(instances)}] Running {issue_id}...")
        t0 = time.time()
        issue_id, result = run_instance(entry, api_key, args.model)
        elapsed = time.time() - t0

        status = "OK" if result["success"] else f"FAIL: {result['error']}"
        pred_count = len(result["predicted_files"])
        gt_count = len(result["ground_truth_files"])
        print(f"  {status} | predicted={pred_count} gt={gt_count} | {elapsed:.1f}s")
        if result["predicted_files"]:
            print(f"  predicted: {result['predicted_files']}")

        # Save per-instance result
        result_file.write_text(json.dumps(result, indent=2))
        all_results[issue_id] = result

    # Compute and save aggregate metrics
    metrics = compute_metrics(all_results)
    summary_file = OUTPUT_DIR / "results.json"
    summary_file.write_text(json.dumps(metrics, indent=2))

    print(f"\n=== Results ({len(all_results)} instances) ===")
    if "error" not in metrics:
        for k, v in metrics["metrics"].items():
            print(f"  {k}: {v}")
    else:
        print(f"  {metrics['error']}")
    print(f"Saved to {summary_file}")


if __name__ == "__main__":
    main()
