#!/usr/bin/env python3
"""
Evaluate acc@1 and acc@5 for locagent outputs.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def compute_acc_at_k(ground_truth: List[str], predicted: List[str], k: int) -> int:
    """
    Check if all ground truth files are found in top-k predictions.
    Returns 1 if all found, 0 otherwise.
    """
    top_k_preds = set(predicted[:k])
    gt_set = set(ground_truth)
    # acc@k: all ground truth files found in top-k
    return 1 if gt_set.issubset(top_k_preds) else 0


def compute_recall_at_k(ground_truth: List[str], predicted: List[str], k: int) -> float:
    """
    Compute recall@k: fraction of ground truth files found in top-k predictions.
    """
    if not ground_truth:
        return 0.0
    top_k_preds = set(predicted[:k])
    gt_set = set(ground_truth)
    found = len(gt_set.intersection(top_k_preds))
    return found / len(gt_set)


def evaluate_folder(folder_path: str) -> Dict[str, Any]:
    """
    Evaluate all JSON files in a folder.
    Returns metrics dict.
    """
    results_dir = os.path.join(folder_path, "results")
    if not os.path.exists(results_dir):
        results_dir = folder_path

    json_files = list(Path(results_dir).glob("*.json"))
    # Exclude results.json if it exists
    json_files = [f for f in json_files if f.name != "results.json"]

    if not json_files:
        return {"error": "No JSON files found", "folder": folder_path}

    total = 0
    acc_1_sum = 0
    acc_5_sum = 0
    recall_1_sum = 0.0
    recall_5_sum = 0.0
    successful = 0
    failed = 0

    details = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if not data.get("success", False):
                failed += 1
                continue

            ground_truth = data.get("ground_truth_files", [])
            predicted = data.get("predicted_files", [])

            if not ground_truth:
                continue

            total += 1
            successful += 1

            acc_1 = compute_acc_at_k(ground_truth, predicted, 1)
            acc_5 = compute_acc_at_k(ground_truth, predicted, 5)
            recall_1 = compute_recall_at_k(ground_truth, predicted, 1)
            recall_5 = compute_recall_at_k(ground_truth, predicted, 5)

            acc_1_sum += acc_1
            acc_5_sum += acc_5
            recall_1_sum += recall_1
            recall_5_sum += recall_5

            details.append({
                "file": json_file.name,
                "acc@1": acc_1,
                "acc@5": acc_5,
                "recall@1": round(recall_1, 4),
                "recall@5": round(recall_5, 4),
                "ground_truth_count": len(ground_truth),
                "predicted_count": len(predicted)
            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    if total == 0:
        return {"error": "No valid samples found", "folder": folder_path}

    results = {
        "folder": folder_path,
        "total_samples": total,
        "successful_runs": successful,
        "failed_runs": failed,
        "metrics": {
            "acc@1": round(acc_1_sum / total, 4),
            "acc@5": round(acc_5_sum / total, 4),
            "recall@1": round(recall_1_sum / total, 4),
            "recall@5": round(recall_5_sum / total, 4),
        },
        "raw_counts": {
            "acc@1_correct": acc_1_sum,
            "acc@5_correct": acc_5_sum,
        }
    }

    return results


def main():
    output_base = "/home/ubuntu/locagent/output"
    folders = ["codeagent", "codeagent_top5", "kg_codeagent_top5"]

    all_results = {}

    for folder in folders:
        folder_path = os.path.join(output_base, folder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        print(f"\nEvaluating {folder}...")
        results = evaluate_folder(folder_path)
        all_results[folder] = results

        # Save results to each folder
        results_file = os.path.join(folder_path, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")

        # Print summary
        if "error" not in results:
            print(f"  Total samples: {results['total_samples']}")
            print(f"  Acc@1: {results['metrics']['acc@1']}")
            print(f"  Acc@5: {results['metrics']['acc@5']}")
            print(f"  Recall@1: {results['metrics']['recall@1']}")
            print(f"  Recall@5: {results['metrics']['recall@5']}")
        else:
            print(f"  Error: {results['error']}")

    # Save combined results
    combined_file = os.path.join(output_base, "all_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {combined_file}")


if __name__ == "__main__":
    main()
