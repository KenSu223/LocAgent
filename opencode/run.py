#!/usr/bin/env python3
"""Run the OpenCode agent for file localization on MULocBench."""

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
OUTPUT_DIR = PROJECT_ROOT / "output" / "opencode"
RESULTS_DIR = OUTPUT_DIR / "results"
OPENCODE_BIN = Path("/home/ubuntu/.opencode/bin/opencode")

AZURE_BASE_URL = "https://cielara-research-resource.cognitiveservices.azure.com/openai"

from opencode.prompt import build_prompt
from opencode.parse_output import parse_opencode_output


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
    """Derive a short identifier for the issue."""
    url = entry.get("iss_html_url", "")
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
    """Get the Azure OpenAI API key from .env file."""
    key = os.environ.get("AZURE_OPENAI_API_KEY_MAIN", "")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("AZURE_OPENAI_API_KEY_MAIN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    return ""


def setup_opencode_config(repo_dir: Path, api_key: str) -> None:
    """Write an opencode.jsonc config inside the repo's .opencode dir."""
    config_dir = repo_dir / ".opencode"
    config_dir.mkdir(exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "azure": {
                "options": {
                    "apiKey": api_key,
                    "baseURL": AZURE_BASE_URL,
                    "useCompletionUrls": True,
                },
            },
        },
    }
    config_file = config_dir / "opencode.jsonc"
    config_file.write_text(json.dumps(config, indent=2))


def run_opencode(repo_dir: Path, prompt: str, model: str = "azure/gpt-5.2") -> str:
    """Invoke opencode run and return raw output."""
    cmd = [
        str(OPENCODE_BIN),
        "run",
        "--format", "json",
        "-m", model,
        prompt,
    ]

    env = os.environ.copy()
    # Ensure opencode doesn't try to use a terminal
    env["NO_COLOR"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        env=env,
        timeout=900,
    )

    output = result.stdout or ""
    if result.returncode != 0 and not output:
        stderr = (result.stderr or "")[:500]
        raise RuntimeError(
            f"opencode run failed (rc={result.returncode}): {stderr}"
        )
    return output


def run_instance(
    entry: Dict[str, Any], api_key: str, model: str,
) -> tuple:
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
        "elapsed_seconds": 0.0,
    }

    t0 = time.time()
    try:
        repo_dir = checkout_repo(repo_name, base_commit)
        setup_opencode_config(repo_dir, api_key)

        user_prompt = build_prompt(title, body)
        raw_output = run_opencode(repo_dir, user_prompt, model)
        result["raw_output"] = raw_output

        predicted = parse_opencode_output(raw_output)
        result["predicted_files"] = predicted[:5]
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["elapsed_seconds"] = round(time.time() - t0, 1)
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

        if gt_set.issubset(set(pred[:1])):
            acc1 += 1
        if gt_set.issubset(set(pred[:5])):
            acc5 += 1
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
    parser = argparse.ArgumentParser(
        description="Run OpenCode agent for file localization",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max instances to run (0=all)",
    )
    parser.add_argument(
        "--repo", type=str, default=None, help="Filter to a specific repo",
    )
    parser.add_argument(
        "--model", type=str, default="azure/gpt-5.2",
        help="Model in provider/model format",
    )
    args = parser.parse_args()

    if not OPENCODE_BIN.exists():
        print(f"ERROR: opencode binary not found at {OPENCODE_BIN}")
        sys.exit(1)

    api_key = get_api_key()
    if not api_key:
        print(
            "ERROR: No API key found. Set AZURE_OPENAI_API_KEY_MAIN env var "
            "or add it to LocAgent/.env"
        )
        sys.exit(1)

    instances = load_dataset(repo_filter=args.repo)
    if args.limit > 0:
        instances = instances[: args.limit]

    print(f"Running {len(instances)} instances (model={args.model})")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for i, entry in enumerate(instances):
        issue_id = get_issue_id(entry)
        result_file = RESULTS_DIR / f"{issue_id}.json"

        # Skip if already completed successfully
        if result_file.exists():
            try:
                existing = json.loads(result_file.read_text())
                if existing.get("success"):
                    print(
                        f"[{i+1}/{len(instances)}] SKIP {issue_id} (already done)"
                    )
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

        result_file.write_text(json.dumps(result, indent=2))
        all_results[issue_id] = result

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
