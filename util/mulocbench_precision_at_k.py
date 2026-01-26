#!/usr/bin/env python3
"""
Compute Precision@k for MULocBench results.json using LocBench metrics.

Usage: python util/mulocbench_precision_at_k.py \
  --results output/kg_codeagent_top5/results/results.json \
  --k 1,3,5,10
"""

import argparse
import json
from pathlib import Path

import torch

from evaluation.eval_metric import acc_at_k, precision_at_k, recall_at_k


def build_pred_labels(results, max_k):
    pred_labels = []
    for item in results:
        gt = set(item.get("ground_truth_files", []))
        preds = item.get("predicted_files", [])[:max_k]
        labels = [1 if p in gt else 0 for p in preds]
        # Pad to max_k to match eval_metric precision_at_k behavior.
        labels.extend([0] * (max_k - len(labels)))
        pred_labels.append(labels)
    return torch.tensor(pred_labels)


def build_gt_labels(results, max_k):
    gt_labels = []
    for item in results:
        gt_len = len(item.get("ground_truth_files", []))
        labels = [1 if i < gt_len else 0 for i in range(max_k)]
        gt_labels.append(labels)
    return torch.tensor(gt_labels)


def compute_metrics_at_k(results, k_values):
    max_k = max(k_values)
    pred_target = build_pred_labels(results, max_k)
    ideal_target = build_gt_labels(results, max_k)
    metrics = {}
    for k in k_values:
        metrics[f"Acc@{k}"] = round(acc_at_k(pred_target, ideal_target, k=k).item(), 4)
        metrics[f"Precision@{k}"] = round(precision_at_k(pred_target, ideal_target, k=k).item(), 4)
        metrics[f"Recall@{k}"] = round(recall_at_k(pred_target, ideal_target, k=k).item(), 4)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Precision@k for MULocBench results.json")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated k values")
    args = parser.parse_args()

    results_path = Path(args.results)
    with results_path.open() as f:
        data = json.load(f)
    results = data.get("results", [])

    k_values = [int(v.strip()) for v in args.k.split(",") if v.strip()]
    metrics = compute_metrics_at_k(results, k_values)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
