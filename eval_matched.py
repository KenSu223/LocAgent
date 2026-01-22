#!/usr/bin/env python3
"""
Compare outputs for instances that appear in both codeagent_top5 and kg_codeagent_top5.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set


def compute_acc_at_k(ground_truth: List[str], predicted: List[str], k: int) -> int:
    """Check if all ground truth files are found in top-k predictions."""
    top_k_preds = set(predicted[:k])
    gt_set = set(ground_truth)
    return 1 if gt_set.issubset(top_k_preds) else 0


def compute_recall_at_k(ground_truth: List[str], predicted: List[str], k: int) -> float:
    """Compute recall@k: fraction of ground truth files found in top-k predictions."""
    if not ground_truth:
        return 0.0
    top_k_preds = set(predicted[:k])
    gt_set = set(ground_truth)
    found = len(gt_set.intersection(top_k_preds))
    return found / len(gt_set)


def get_json_files(folder: str) -> Dict[str, Path]:
    """Get dict of filename -> path for all JSON files in folder."""
    results_dir = os.path.join(folder, "results")
    json_files = {}
    for f in Path(results_dir).glob("*.json"):
        if f.name != "results.json":
            json_files[f.name] = f
    return json_files


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_instance(data: Dict) -> Dict:
    """Evaluate a single instance."""
    if not data.get("success", False):
        return None

    ground_truth = data.get("ground_truth_files", [])
    predicted = data.get("predicted_files", [])

    if not ground_truth:
        return None

    return {
        "acc@1": compute_acc_at_k(ground_truth, predicted, 1),
        "acc@5": compute_acc_at_k(ground_truth, predicted, 5),
        "acc@10": compute_acc_at_k(ground_truth, predicted, 10),
        "recall@1": round(compute_recall_at_k(ground_truth, predicted, 1), 4),
        "recall@5": round(compute_recall_at_k(ground_truth, predicted, 5), 4),
        "recall@10": round(compute_recall_at_k(ground_truth, predicted, 10), 4),
        "ground_truth_count": len(ground_truth),
        "predicted_count": len(predicted),
    }


def main():
    output_base = "/home/ubuntu/locagent/output"
    folder1 = os.path.join(output_base, "codeagent_top5")
    folder2 = os.path.join(output_base, "kg_codeagent_top5")

    files1 = get_json_files(folder1)
    files2 = get_json_files(folder2)

    # Find matching files
    matching_files = set(files1.keys()) & set(files2.keys())
    print(f"Found {len(matching_files)} matching instances")

    # Evaluate both
    codeagent_metrics = {"acc@1": 0, "acc@5": 0, "acc@10": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}
    kg_metrics = {"acc@1": 0, "acc@5": 0, "acc@10": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}

    valid_count = 0
    instance_details = []

    for filename in sorted(matching_files):
        data1 = load_json(files1[filename])
        data2 = load_json(files2[filename])

        eval1 = evaluate_instance(data1)
        eval2 = evaluate_instance(data2)

        # Both must be valid
        if eval1 is None or eval2 is None:
            continue

        valid_count += 1

        for key in ["acc@1", "acc@5", "acc@10", "recall@1", "recall@5", "recall@10"]:
            codeagent_metrics[key] += eval1[key]
            kg_metrics[key] += eval2[key]

        instance_details.append({
            "instance": filename.replace(".json", ""),
            "ground_truth_files": data1.get("ground_truth_files", []),
            "codeagent_top5": {
                "predicted_files": data1.get("predicted_files", []),
                **eval1
            },
            "kg_codeagent_top5": {
                "predicted_files": data2.get("predicted_files", []),
                **eval2
            }
        })

    print(f"Valid matched instances: {valid_count}")

    # Compute averages
    for key in codeagent_metrics:
        codeagent_metrics[key] = round(codeagent_metrics[key] / valid_count, 4)
        kg_metrics[key] = round(kg_metrics[key] / valid_count, 4)

    results = {
        "summary": {
            "matched_instances": len(matching_files),
            "valid_instances": valid_count,
            "codeagent_top5": codeagent_metrics,
            "kg_codeagent_top5": kg_metrics,
        },
        "comparison": {
            "acc@1_diff": round(kg_metrics["acc@1"] - codeagent_metrics["acc@1"], 4),
            "acc@5_diff": round(kg_metrics["acc@5"] - codeagent_metrics["acc@5"], 4),
            "acc@10_diff": round(kg_metrics["acc@10"] - codeagent_metrics["acc@10"], 4),
            "recall@1_diff": round(kg_metrics["recall@1"] - codeagent_metrics["recall@1"], 4),
            "recall@5_diff": round(kg_metrics["recall@5"] - codeagent_metrics["recall@5"], 4),
            "recall@10_diff": round(kg_metrics["recall@10"] - codeagent_metrics["recall@10"], 4),
        },
        "instances": instance_details
    }

    # Save results
    output_file = os.path.join(output_base, "matched_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {valid_count} matched instances:")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'codeagent_top5':>16} {'kg_codeagent_top5':>18} {'Diff':>10}")
    print(f"{'-'*60}")
    for key in ["acc@1", "acc@5", "acc@10", "recall@1", "recall@5", "recall@10"]:
        diff = results["comparison"][f"{key}_diff"]
        sign = "+" if diff > 0 else ""
        print(f"{key:<12} {codeagent_metrics[key]:>16.4f} {kg_metrics[key]:>18.4f} {sign}{diff:>9.4f}")


if __name__ == "__main__":
    main()
