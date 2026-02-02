"""
Usage:
python scripts/evaluate_mulocbench.py \
  --loc_file evaluation_results/mulocbench_subset/location/loc_outputs.jsonl \
  --gt_file evaluation_results/mulocbench_subset/gt_location.jsonl \
  --output_csv evaluation_results/mulocbench_subset/evaluation_scores.csv \
  --repo flask \
  --only_present \
  --skip_empty

  Make sure GT locations match with the target repo for evalution

"""


import argparse
import sys
import json

import pandas as pd

sys.path.append(".")

from evaluation.eval_metric import eval_w_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loc_file",
        type=str,
        required=True,
        help="Path to loc_outputs.jsonl (or merged_loc_outputs_*.jsonl).",
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to gt_location.jsonl produced by convert_mulocbench.py.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to write evaluation_scores.csv.",
    )
    parser.add_argument(
        "--only_present",
        action="store_true",
        help="Evaluate only instance_ids present in the localization output file.",
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="When used with --only_present, ignore entries with empty found_files.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Evaluate only a single repo (e.g., 'pallets/flask' or 'flask').",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo list (e.g., 'pallets/flask,flask,psf/requests').",
    )
    args = parser.parse_args()

    level2key_dict = {
        "file": "found_files",
        "module": "found_modules",
        "function": "found_entities",
    }

    selected_list = None
    repo_filter = None
    if args.repo or args.repos:
        repo_filter = set()
        repo_short_filter = set()
        if args.repo:
            repo_value = args.repo.strip()
            repo_filter.add(repo_value)
            if "/" not in repo_value:
                repo_short_filter.add(repo_value)
        if args.repos:
            for repo_value in [r.strip() for r in args.repos.split(",") if r.strip()]:
                repo_filter.add(repo_value)
                if "/" not in repo_value:
                    repo_short_filter.add(repo_value)
        repo_selected = []
        with open(args.gt_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                repo_full = record.get("repo")
                repo_short = repo_full.split("/")[-1] if repo_full else None
                if repo_full in repo_filter or repo_short in repo_short_filter:
                    repo_selected.append(record["instance_id"])
        selected_list = repo_selected

    if args.only_present:
        present_list = []
        with open(args.loc_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if args.skip_empty:
                    found_files = record.get("found_files", [])
                    if found_files == [[]] or found_files == []:
                        continue
                present_list.append(record["instance_id"])
        if selected_list is None:
            selected_list = present_list
        else:
            selected_set = set(selected_list)
            selected_list = [instance_id for instance_id in present_list if instance_id in selected_set]

    results = eval_w_file(
        args.gt_file,
        args.loc_file,
        level2key_dict,
        selected_list=selected_list,
    )
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string())
    print("\n")

    if args.output_csv:
        results.to_csv(args.output_csv)
        print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
