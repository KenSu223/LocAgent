"""
Evaluate Cursor CLI results using LocAgent's evaluation metrics.

Usage:
    # Evaluate only instances present in your output file
    python run_locagent_eval.py \
        --loc_file /path/to/loc_outputs.jsonl \
        --gt_file /path/to/mulocbench.jsonl \
        --only_predicted

    # Evaluate on specific repos
    python run_locagent_eval.py \
        --loc_file /path/to/loc_outputs.jsonl \
        --gt_file /path/to/mulocbench.jsonl \
        --repos scikit-learn flask requests transformers pandas
    
    # Evaluate on specific repos, only for instances you have predictions for
    python run_locagent_eval.py \
        --loc_file /path/to/loc_outputs.jsonl \
        --gt_file /path/to/mulocbench.jsonl \
        --repos flask requests \
        --only_predicted

    python run_locagent_eval.py \
        --loc_file /home/tsu25/LocAgent/results/codex_agent/loc_outputs_filtered.jsonl \
        --gt_file /home/tsu25/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
        --only_predicted

     /home/tsu25/LocAgent/results/codex_agent/loc_outputs_filtered.jsonl
"""

import argparse
import json
import sys
import collections
import re
import torch
import pandas as pd
from typing import Optional, List
from torch import Tensor


# ============== Metric Functions (from LocAgent) ==============

def _dcg(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target / torch.log2(rank_positions + 1)).sum(dim=-1)


def div_no_nan(a: Tensor, b: Tensor, na_value: Optional[float] = 0.) -> Tensor:
    return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)


def normalized_dcg(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    ideal_target = ideal_target[:, :k]
    return div_no_nan(_dcg(pred_target), _dcg(ideal_target)).mean(0)


def recall_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    relevant = (pred_target == 1).sum(dim=-1)
    total_relevant = (ideal_target == 1).sum(dim=-1)
    recall = div_no_nan(relevant, total_relevant, na_value=0.)
    return recall.mean(0)


def acc_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    ideal_target = ideal_target[:, :k]
    
    relevant = (pred_target == 1).sum(dim=-1)
    total_relevant = (ideal_target == 1).sum(dim=-1)

    comparison = relevant == total_relevant
    return comparison.sum()/relevant.shape[0]


def precision_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    relevant = (pred_target == 1).sum(dim=-1)
    precision = relevant / k
    return precision.mean(0)


METRIC_FUNC = {
    'ndcg': normalized_dcg,
    'recall': recall_at_k,
    'acc': acc_at_k,
    'precision': precision_at_k,
}
METRIC_NAME = {
    'ndcg': 'NDCG',
    'recall': 'Recall',
    'acc': 'Acc',
    'precision': 'P',
}


# ============== Helper Functions ==============

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def convert_solutions_dict(dataset, key='model_patch'):
    return {elem['instance_id']: elem.get(key, []) for elem in dataset}


def extract_repo_from_instance_id(instance_id: str) -> str:
    """
    Extract repo name from instance_id.
    Common formats:
      - "owner__repo__issue_number" (e.g., "pallets__flask__4992")
      - "repo__issue_number" (e.g., "flask__4992")
    Returns normalized repo name (lowercase, hyphens replaced with underscores).
    """
    parts = instance_id.split('__')
    if len(parts) >= 2:
        # Could be owner__repo__number or repo__number
        # Check if second part looks like a number
        if len(parts) >= 3 and parts[2].isdigit():
            repo_name = parts[1]  # owner__repo__number format
        elif parts[1].isdigit():
            repo_name = parts[0]  # repo__number format
        else:
            repo_name = parts[1]  # Assume owner__repo format
    else:
        repo_name = parts[0]
    
    return repo_name.lower().replace('-', '_')


def normalize_repo_name(name: str) -> str:
    """Normalize repo name for comparison."""
    return name.lower().replace('-', '_').replace(' ', '_')


def repo_matches(repo_name: str, filter_terms: List[str]) -> bool:
    """Check if repo_name matches any of the filter terms."""
    repo_normalized = normalize_repo_name(repo_name)
    for term in filter_terms:
        term_normalized = normalize_repo_name(term)
        # Exact match or substring match
        if term_normalized == repo_normalized:
            return True
        if term_normalized in repo_normalized or repo_normalized in term_normalized:
            return True
    return False


# ============== Main Evaluation Function ==============

def cal_metrics_w_mulocbench(
    gt_file: str,
    loc_file: str,
    key: str,
    eval_level: str,
    k_values: List[int],
    metrics: List[str] = ['acc'],
    selected_list: List[str] = None,
):
    """
    Calculate metrics using MULocBench format ground truth.
    
    Ground truth is in 'edit_functions' field with format: ["file.py:function", "file.py:Class.method"]
    """
    assert key in ['found_files', 'found_modules', 'found_entities']
    max_k = max(k_values)
    
    # Load ground truth from MULocBench file
    gt_data = load_jsonl(gt_file)
    gt_dict = collections.defaultdict(list)
    
    for instance in gt_data:
        instance_id = instance['instance_id']
        edit_functions = instance.get('edit_functions', [])
        
        if eval_level == 'file':
            # Extract file paths: "flask/cli.py:find_best_app" -> "flask/cli.py"
            for func in edit_functions:
                fn = func.split(':')[0]
                if fn not in gt_dict[instance_id]:
                    gt_dict[instance_id].append(fn)
        
        elif eval_level == 'module':
            # Extract module (class level): "file.py:Class.method" -> "file.py:Class"
            for func in edit_functions:
                parts = func.split(':')
                if len(parts) >= 2:
                    fn = parts[0]
                    func_part = parts[1]
                    if '.' in func_part:
                        # Class.method -> Class
                        class_name = func_part.split('.')[0]
                        mid = f'{fn}:{class_name}'
                    else:
                        # Top-level function
                        mid = func
                    if mid not in gt_dict[instance_id]:
                        gt_dict[instance_id].append(mid)
        
        elif eval_level == 'function':
            # Use full function path
            for func in edit_functions:
                # Handle __init__ suffix
                if func.endswith('.__init__'):
                    func = func[:-len('.__init__')]
                if func not in gt_dict[instance_id]:
                    gt_dict[instance_id].append(func)
    
    # Load predictions
    pred_dict = convert_solutions_dict(load_jsonl(loc_file), key=key)
    
    # Process predictions for module level if needed
    if eval_level == 'module' and key == 'found_entities':
        for ins in pred_dict:
            pred_funcs = pred_dict[ins]
            pred_modules = []
            for pf in pred_funcs:
                parts = pf.split(':')
                if len(parts) >= 2:
                    fn = parts[0]
                    func_part = parts[1]
                    if '.' in func_part:
                        class_name = func_part.split('.')[0]
                        module_loc = f'{fn}:{class_name}'
                    else:
                        module_loc = pf
                    if module_loc not in pred_modules:
                        pred_modules.append(module_loc)
            pred_dict[ins] = pred_modules
    
    # Build label tensors
    _gt_labels = []
    _pred_labels = []
    
    for instance_id in gt_dict.keys():
        if selected_list is not None and instance_id not in selected_list:
            continue
        if not gt_dict[instance_id]:
            continue
        
        if instance_id not in pred_dict:
            pred_locs = []
        else:
            pred_locs = pred_dict[instance_id][:max_k]
        
        gt_labels = [0 for _ in range(max_k)]
        pred_labels = [0 for _ in range(max_k)]
        
        for i in range(len(gt_dict[instance_id])):
            if i < max_k:
                gt_labels[i] = 1
        
        for i, loc in enumerate(pred_locs):
            if loc in gt_dict[instance_id]:
                pred_labels[i] = 1
        
        _gt_labels.append(gt_labels)
        _pred_labels.append(pred_labels)
    
    if not _pred_labels:
        print(f"Warning: No instances to evaluate for {eval_level} level")
        return {}
    
    _pred_target = torch.tensor(_pred_labels)
    _ideal_target = torch.tensor(_gt_labels)
    
    result = {}
    for metric in metrics:
        assert metric in METRIC_FUNC.keys(), f"Unknown metric: {metric}"
        
        metric_func = METRIC_FUNC[metric]
        name = METRIC_NAME[metric]
        for k in k_values:
            value = metric_func(_pred_target, _ideal_target, k=k)
            result[f'{name}@{k}'] = round(value.item(), 4)
    
    return result


def evaluate_mulocbench(
    gt_file: str,
    loc_file: str,
    selected_list: List[str] = None,
    metrics: List[str] = ['acc'],
    k_values_list: List[List[int]] = None,
):
    """
    Evaluate localization results against MULocBench ground truth.
    """
    if not k_values_list:
        k_values_list = [
            [1, 3, 5, 10],   # file level
            [1, 3, 5, 10],     # module level
            [1, 3, 5, 10]      # function level
        ]
    
    file_res = cal_metrics_w_mulocbench(
        gt_file, loc_file,
        key='found_files',
        eval_level='file',
        k_values=k_values_list[0],
        metrics=metrics,
        selected_list=selected_list
    )
    
    module_res = cal_metrics_w_mulocbench(
        gt_file, loc_file,
        key='found_modules',
        eval_level='module',
        k_values=k_values_list[1],
        metrics=metrics,
        selected_list=selected_list
    )
    
    function_res = cal_metrics_w_mulocbench(
        gt_file, loc_file,
        key='found_entities',
        eval_level='function',
        k_values=k_values_list[2],
        metrics=metrics,
        selected_list=selected_list
    )
    
    all_df = pd.concat(
        [pd.DataFrame(res, index=[0]) for res in [file_res, module_res, function_res]],
        axis=1,
        keys=['file', 'module', 'function']
    )
    
    return all_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate localization results using LocAgent metrics")
    parser.add_argument(
        "--loc_file",
        type=str,
        required=True,
        help="Path to loc_outputs.jsonl file (predictions)"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default="/Users/kensu/Downloads/loc_agent/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl",
        help="Path to ground truth MULocBench file"
    )
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="Filter evaluation to specific repos (e.g., --repos scikit-learn flask)"
    )
    parser.add_argument(
        "--only_predicted",
        action="store_true",
        help="Only evaluate instances that are present in the loc_file"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["acc", "recall", "precision"],
        help="Metrics to compute (default: acc recall precision). Options: acc, ndcg, precision, recall"
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include instances with empty predictions (default: skip them)"
    )
    
    args = parser.parse_args()
    
    # Build selected_list based on filters
    selected_list = None
    predicted_instances = None
    
    # If --only_predicted, extract instance IDs from loc_file (excluding empty predictions unless --include_empty)
    if args.only_predicted:
        print(f"Loading instance IDs from: {args.loc_file}")
        predicted_instances = set()
        empty_count = 0
        with open(args.loc_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Check if this instance has any actual predictions
                    has_predictions = False
                    for key in ['found_files', 'found_modules', 'found_entities']:
                        if key in data and data[key] and any(item for item in data[key] if item):
                            has_predictions = True
                            break
                    
                    if has_predictions:
                        predicted_instances.add(data['instance_id'])
                    elif args.include_empty:
                        predicted_instances.add(data['instance_id'])
                        empty_count += 1
                    else:
                        empty_count += 1
        
        print(f"Found {len(predicted_instances)} instances with predictions in loc_file")
        if empty_count > 0:
            if args.include_empty:
                print(f"  Included {empty_count} empty/failed instances (--include_empty)")
            else:
                print(f"  Skipped {empty_count} empty/failed instances")
    
    # If --repos filter is specified
    if args.repos:
        print(f"Filtering evaluation to repos: {args.repos}")
        gt_data = load_jsonl(args.gt_file)
        
        filter_terms = args.repos
        
        repo_filtered_list = []
        for instance in gt_data:
            instance_id = instance['instance_id']
            
            # Try to get repo from 'repo' field first, fallback to parsing instance_id
            if 'repo' in instance:
                # repo field format: "owner/repo_name"
                repo_name = instance['repo'].split('/')[-1]
            else:
                repo_name = extract_repo_from_instance_id(instance_id)
            
            # Check if repo matches any filter term
            if repo_matches(repo_name, filter_terms):
                repo_filtered_list.append(instance_id)
        
        print(f"Found {len(repo_filtered_list)} instances matching repos filter")
        
        # Show breakdown by matched repos
        if repo_filtered_list:
            repo_counts = collections.Counter()
            for iid in repo_filtered_list:
                repo_name = extract_repo_from_instance_id(iid)
                repo_counts[repo_name] += 1
            print(f"  Breakdown: {dict(repo_counts)}")
        
        # Combine with --only_predicted if both are specified
        if predicted_instances is not None:
            selected_list = [iid for iid in repo_filtered_list if iid in predicted_instances]
            print(f"After intersecting with predicted instances: {len(selected_list)} instances")
        else:
            selected_list = repo_filtered_list
    
    elif predicted_instances is not None:
        # Only --only_predicted was specified (no --repos)
        selected_list = list(predicted_instances)
    
    # Run evaluation
    print(f"\nEvaluating:")
    print(f"  Predictions: {args.loc_file}")
    print(f"  Ground truth: {args.gt_file}")
    print(f"  Metrics: {args.metrics}")
    if selected_list is not None:
        print(f"  Evaluating {len(selected_list)} instances")
        if len(selected_list) == 0:
            print("\nNo instances to evaluate. Exiting.")
            sys.exit(0)
    print("-" * 60)
    
    results = evaluate_mulocbench(
        gt_file=args.gt_file,
        loc_file=args.loc_file,
        selected_list=selected_list,
        metrics=args.metrics,
    )
    
    print("\nRESULTS:")
    print("=" * 60)
    print(results.to_string())
    print("=" * 60)
    
    # Also save to CSV
    output_csv = args.loc_file.replace('.jsonl', '_eval_results.csv')
    results.to_csv(output_csv)
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
