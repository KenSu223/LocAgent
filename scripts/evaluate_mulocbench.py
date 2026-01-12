import argparse
import sys

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
    args = parser.parse_args()

    level2key_dict = {
        "file": "found_files",
        "module": "found_modules",
        "function": "found_entities",
    }

    results = eval_w_file(args.gt_file, args.loc_file, level2key_dict)
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
