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
    args = parser.parse_args()

    level2key_dict = {
        "file": "found_files",
        "module": "found_modules",
        "function": "found_entities",
    }

    selected_list = None
    if args.only_present:
        selected_list = []
        with open(args.loc_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if args.skip_empty:
                    found_files = record.get("found_files", [])
                    if found_files == [[]] or found_files == []:
                        continue
                selected_list.append(record["instance_id"])

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
