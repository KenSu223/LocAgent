"""
Evaluate Results with LocAgent Metrics

This script takes the output from evaluate_mulocbench.py and computes
metrics compatible with LocAgent's evaluation pipeline.

Usage:
    python evaluate_with_locagent_metrics.py \
        --predictions results/cursor_evaluation/loc_outputs.jsonl \
        --ground_truth evaluation/datasets/mulocbench/mulocbench.jsonl \
        --output_dir results/cursor_evaluation/metrics
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_file_from_location(loc: str) -> str:
    """Extract file path from location string like 'file.py:func'"""
    return loc.split(":")[0] if ":" in loc else loc


def extract_module_from_location(loc: str) -> str:
    """
    Extract module (class) from location string.
    'file.py:Class.method' -> 'file.py:Class'
    'file.py:func' -> 'file.py:func' (standalone function treated as module)
    """
    if ":" not in loc:
        return loc
    
    file_part, entity = loc.split(":", 1)
    if "." in entity:
        # Class.method -> Class
        class_name = entity.split(".")[0]
        return f"{file_part}:{class_name}"
    else:
        # Standalone function
        return loc


def compute_accuracy_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int
) -> bool:
    """
    Compute Acc@k: whether ALL ground truth locations are in top-k predictions.
    
    Following LocAgent's evaluation: a localization is successful only if
    ALL relevant locations are correctly identified within top-k predictions.
    """
    if not ground_truth:
        return True  # No ground truth = trivially correct
    
    gt_set = set(ground_truth)
    pred_set = set(predictions[:k])
    
    return gt_set.issubset(pred_set)


def compute_precision_recall_f1(
    predictions: List[str],
    ground_truth: List[str]
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 score"""
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predictions:
        return 0.0, 0.0, 0.0
    if not ground_truth:
        return 0.0, 1.0, 0.0
    
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    
    true_positives = len(pred_set & gt_set)
    
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def evaluate_file_level(
    pred_locs: List[str],
    gt_locs: List[str]
) -> Dict:
    """Evaluate at file level"""
    pred_files = [extract_file_from_location(loc) for loc in pred_locs]
    gt_files = list(set(extract_file_from_location(loc) for loc in gt_locs))
    
    return {
        "acc@1": compute_accuracy_at_k(pred_files, gt_files, 1),
        "acc@3": compute_accuracy_at_k(pred_files, gt_files, 3),
        "acc@5": compute_accuracy_at_k(pred_files, gt_files, 5),
        "precision_recall_f1": compute_precision_recall_f1(pred_files, gt_files)
    }


def evaluate_module_level(
    pred_locs: List[str],
    gt_locs: List[str]
) -> Dict:
    """
    Evaluate at module (class) level.
    A prediction is correct if it identifies any function within the patched class.
    """
    pred_modules = [extract_module_from_location(loc) for loc in pred_locs]
    gt_modules = list(set(extract_module_from_location(loc) for loc in gt_locs))
    
    return {
        "acc@5": compute_accuracy_at_k(pred_modules, gt_modules, 5),
        "acc@10": compute_accuracy_at_k(pred_modules, gt_modules, 10),
        "precision_recall_f1": compute_precision_recall_f1(pred_modules, gt_modules)
    }


def evaluate_function_level(
    pred_locs: List[str],
    gt_locs: List[str]
) -> Dict:
    """Evaluate at function level (exact match)"""
    return {
        "acc@5": compute_accuracy_at_k(pred_locs, gt_locs, 5),
        "acc@10": compute_accuracy_at_k(pred_locs, gt_locs, 10),
        "precision_recall_f1": compute_precision_recall_f1(pred_locs, gt_locs)
    }


def evaluate_results(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Dict:
    """
    Evaluate predictions against ground truth.
    
    This follows LocAgent's evaluation methodology:
    - Acc@k: All ground truth locations found in top-k predictions
    - Evaluated at file, module (class), and function levels
    """
    
    # Build ground truth lookup
    gt_lookup = {r["instance_id"]: r for r in ground_truth}
    
    # Aggregate metrics
    results = {
        "total": 0,
        "evaluated": 0,
        "skipped": 0,
        "file_level": defaultdict(int),
        "module_level": defaultdict(int),
        "function_level": defaultdict(int),
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    detailed_results = []
    
    for pred in predictions:
        instance_id = pred["instance_id"]
        results["total"] += 1
        
        # Skip if error
        if pred.get("error"):
            results["skipped"] += 1
            continue
        
        # Find ground truth
        gt = gt_lookup.get(instance_id)
        if not gt:
            logger.warning(f"No ground truth for {instance_id}")
            results["skipped"] += 1
            continue
        
        results["evaluated"] += 1
        
        # Get predictions and ground truth
        pred_files = pred.get("found_files", [])
        pred_functions = pred.get("found_edit_locs", [])
        
        # Combine files and functions for comprehensive predictions
        all_pred_locs = list(set(pred_functions + [f"{f}:" for f in pred_files if ":" not in f]))
        
        gt_functions = gt.get("edit_functions", [])
        
        # Evaluate at each level
        file_eval = evaluate_file_level(pred_functions or pred_files, gt_functions)
        module_eval = evaluate_module_level(pred_functions, gt_functions)
        func_eval = evaluate_function_level(pred_functions, gt_functions)
        
        # Aggregate
        for metric in ["acc@1", "acc@3", "acc@5"]:
            if metric in file_eval:
                results["file_level"][metric] += int(file_eval[metric])
        
        for metric in ["acc@5", "acc@10"]:
            if metric in module_eval:
                results["module_level"][metric] += int(module_eval[metric])
            if metric in func_eval:
                results["function_level"][metric] += int(func_eval[metric])
        
        # Precision/Recall/F1 (function level)
        p, r, f = func_eval["precision_recall_f1"]
        results["precision"].append(p)
        results["recall"].append(r)
        results["f1"].append(f)
        
        # Store detailed result
        detailed_results.append({
            "instance_id": instance_id,
            "file_level": file_eval,
            "module_level": module_eval,
            "function_level": func_eval,
            "pred_files": pred_files,
            "pred_functions": pred_functions,
            "gt_functions": gt_functions
        })
    
    # Compute averages
    n = results["evaluated"]
    if n > 0:
        summary = {
            "total": results["total"],
            "evaluated": results["evaluated"],
            "skipped": results["skipped"],
            "file_level": {
                "acc@1": results["file_level"]["acc@1"] / n * 100,
                "acc@3": results["file_level"]["acc@3"] / n * 100,
                "acc@5": results["file_level"]["acc@5"] / n * 100,
            },
            "module_level": {
                "acc@5": results["module_level"]["acc@5"] / n * 100,
                "acc@10": results["module_level"]["acc@10"] / n * 100,
            },
            "function_level": {
                "acc@5": results["function_level"]["acc@5"] / n * 100,
                "acc@10": results["function_level"]["acc@10"] / n * 100,
            },
            "avg_precision": sum(results["precision"]) / n * 100,
            "avg_recall": sum(results["recall"]) / n * 100,
            "avg_f1": sum(results["f1"]) / n * 100,
        }
    else:
        summary = {"error": "No valid predictions to evaluate"}
    
    return summary, detailed_results


def print_metrics(metrics: Dict):
    """Print metrics in a formatted table"""
    print("\n" + "="*60)
    print("LOCAGENT-COMPATIBLE EVALUATION METRICS")
    print("="*60)
    
    print(f"\nDataset Statistics:")
    print(f"  Total instances:     {metrics.get('total', 0)}")
    print(f"  Evaluated:           {metrics.get('evaluated', 0)}")
    print(f"  Skipped (errors):    {metrics.get('skipped', 0)}")
    
    if "file_level" in metrics:
        print(f"\nFile-Level Localization:")
        print(f"  Acc@1: {metrics['file_level']['acc@1']:.2f}%")
        print(f"  Acc@3: {metrics['file_level']['acc@3']:.2f}%")
        print(f"  Acc@5: {metrics['file_level']['acc@5']:.2f}%")
    
    if "module_level" in metrics:
        print(f"\nModule-Level (Class) Localization:")
        print(f"  Acc@5:  {metrics['module_level']['acc@5']:.2f}%")
        print(f"  Acc@10: {metrics['module_level']['acc@10']:.2f}%")
    
    if "function_level" in metrics:
        print(f"\nFunction-Level Localization:")
        print(f"  Acc@5:  {metrics['function_level']['acc@5']:.2f}%")
        print(f"  Acc@10: {metrics['function_level']['acc@10']:.2f}%")
    
    if "avg_precision" in metrics:
        print(f"\nFunction-Level Precision/Recall/F1:")
        print(f"  Precision: {metrics['avg_precision']:.2f}%")
        print(f"  Recall:    {metrics['avg_recall']:.2f}%")
        print(f"  F1 Score:  {metrics['avg_f1']:.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code localization results with LocAgent-compatible metrics"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth JSONL file (MULocBench in LocBench format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
        help="Directory to save detailed metrics"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading predictions from {args.predictions}")
    predictions = load_jsonl(args.predictions)
    
    logger.info(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)
    
    # Evaluate
    summary, detailed = evaluate_results(predictions, ground_truth)
    
    # Print results
    print_metrics(summary)
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(args.output_dir) / "summary_metrics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(Path(args.output_dir) / "detailed_results.jsonl", 'w') as f:
        for r in detailed:
            f.write(json.dumps(r) + "\n")
    
    logger.info(f"Saved metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
