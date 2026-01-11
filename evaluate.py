#!/usr/bin/env python3
"""
Evaluation script for LocAgent results.

Usage:
    python evaluate.py --loc_file <path_to_loc_outputs.jsonl> [options]
"""

import argparse
import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append('.')

from evaluation.eval_metric import evaluate_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate LocAgent localization results')
    parser.add_argument(
        '--loc_file',
        type=str,
        required=True,
        help='Path to localization results JSONL file (e.g., ./results/loc_outputs.jsonl)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='czlll/Loc-Bench_V1',
        help='Dataset name (default: czlll/Loc-Bench_V1)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split (default: test)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to save results as CSV (optional)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['acc', 'ndcg', 'precision', 'recall', 'map'],
        choices=['acc', 'ndcg', 'precision', 'recall', 'map'],
        help='Metrics to calculate (default: all)'
    )
    parser.add_argument(
        '--only_loc_instances',
        action='store_true',
        help='Evaluate only the instance_ids present in --loc_file'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.loc_file):
        print(f"Error: File not found: {args.loc_file}")
        sys.exit(1)
    
    # Define level to key mapping
    level2key_dict = {
        'file': 'found_files',
        'module': 'found_modules',
        'function': 'found_entities',
    }
    
    print("="*80)
    print("LocAgent Evaluation")
    print("="*80)
    print(f"Results file: {args.loc_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print("="*80)
    print("\nEvaluating... This may take a few minutes...\n")
    
    try:
        selected_list = None
        if args.only_loc_instances:
            selected_list = []
            with open(args.loc_file, 'r') as f:
                for line in f:
                    selected_list.append(pd.read_json(line, typ='series')["instance_id"])

        # Evaluate results
        results = evaluate_results(
            loc_file=args.loc_file,
            level2key_dict=level2key_dict,
            dataset=args.dataset,
            split=args.split,
            metrics=args.metrics,
            selected_list=selected_list,
            k_values_list=[
                [1, 3, 5],    # File-level: k=1,3,5
                [5, 10],      # Module-level: k=5,10
                [5, 10]       # Function-level: k=5,10
            ]
        )
        
        # Display results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(results.to_string())
        print("="*80)
        
        # Save to CSV if requested
        if args.output_csv:
            results.to_csv(args.output_csv)
            print(f"\n✅ Results saved to: {args.output_csv}")
        else:
            # Auto-save next to input file
            output_csv = args.loc_file.replace('.jsonl', '_evaluation.csv')
            results.to_csv(output_csv)
            print(f"\n✅ Results saved to: {output_csv}")
        
        print("\n" + "="*80)
        print("METRIC EXPLANATIONS")
        print("="*80)
        print("Acc@k:  Accuracy - percentage of instances where ALL correct locations are in top-k")
        print("Recall@k: Percentage of correct locations found in top-k predictions")
        print("NDCG@k:  Normalized Discounted Cumulative Gain - ranking quality (higher is better)")
        print("P@k:     Precision - percentage of top-k predictions that are correct")
        print("MAP@k:   Mean Average Precision - average precision across all instances")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
