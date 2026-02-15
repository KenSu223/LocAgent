#!/usr/bin/env python3
"""
Prepare Codex agent execution data for LLM analysis.

This script creates analysis-ready files that combine:
- Predictions vs ground truth
- Execution traces (reasoning, commands, tool use)
- Success/failure categorization
- Performance metrics per instance

Output formats:
1. analysis_data.jsonl - Full structured data for programmatic analysis
2. analysis_report.md - Human/LLM readable markdown report
3. failure_cases.jsonl - Only failed instances with detailed traces
4. success_cases.jsonl - Successful instances for pattern comparison

Usage:
    python prepare_trace_analysis.py \
        --loc_file results/codex_agent/loc_outputs_filtered.jsonl \
        --gt_file evaluation/mulocbench_2/mulocbench.jsonl \
        --log_file results/codex_agent/evaluation_*.log \
        --output_dir analysis/

python prepare_trace_analysis.py \
    --loc_file /home/tsu25/LocAgent/results/codex_agent/loc_outputs_filtered.jsonl \
    --gt_file /home/tsu25/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --log_file "/home/tsu25/LocAgent/results/codex_agent/recent/*.log" \
    --output_dir /home/tsu25/LocAgent/codex_agent/analysis/
"""

import argparse
import json
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pandas as pd


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_log_for_instance(log_content: str, instance_id: str) -> Dict:
    """Extract execution trace for a specific instance from log in sequential order."""
    # Find the section for this instance
    pattern = rf"Instance: {re.escape(instance_id)}.*?(?=Instance: |\Z)"
    match = re.search(pattern, log_content, re.DOTALL)
    
    if not match:
        return {}
    
    instance_log = match.group(0)
    
    # Extract all trace events in sequential order
    trace_events = []
    
    # Pattern to match all trace components with their positions
    # System errors to ignore (timeout, rate limit, reconnecting)
    ignore_patterns = [
        r'failed to refresh available models',
        r'TIMEOUT after \d+s',
        r'Reconnecting\.\.\.',
        r'stream disconnected',
        r'rate limit',
        r'401 Unauthorized',
    ]
    
    # Find all log lines with timestamps and types
    for line in instance_log.split('\n'):
        # Skip system errors
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in ignore_patterns):
            continue
        
        # Extract reasoning
        reasoning_match = re.search(r'\[REASONING\] (.*)', line)
        if reasoning_match:
            trace_events.append({
                'type': 'reasoning',
                'content': reasoning_match.group(1).strip()
            })
            continue
        
        # Extract commands
        command_match = re.search(r'\[COMMAND\] (.*)', line)
        if command_match:
            trace_events.append({
                'type': 'command',
                'content': command_match.group(1).strip()
            })
            continue
        
        # Extract outputs
        output_match = re.search(r'\[OUTPUT\] (.*)', line)
        if output_match:
            content = output_match.group(1).strip()
            # Truncate very long outputs
            if len(content) > 300:
                content = content[:300] + '...'
            trace_events.append({
                'type': 'output',
                'content': content
            })
            continue
        
        # Extract files found
        file_grep_match = re.search(r'\[FOUND FILE from grep\] (.*)', line)
        if file_grep_match:
            trace_events.append({
                'type': 'found_file_grep',
                'content': file_grep_match.group(1).strip()
            })
            continue
        
        file_response_match = re.search(r'\[FOUND FILE from response\] (.*)', line)
        if file_response_match:
            trace_events.append({
                'type': 'found_file_response',
                'content': file_response_match.group(1).strip()
            })
            continue
        
        # Extract entities found
        entity_match = re.search(r'\[FOUND ENTITY from response\] (.*)', line)
        if entity_match:
            trace_events.append({
                'type': 'found_entity',
                'content': entity_match.group(1).strip()
            })
            continue
        
        # Extract assistant final response
        assistant_match = re.search(r'\[ASSISTANT\] (.*)', line)
        if assistant_match:
            trace_events.append({
                'type': 'assistant_response',
                'content': assistant_match.group(1).strip()
            })
            continue
    
    # Collect summary statistics
    grep_files = [e['content'] for e in trace_events if e['type'] == 'found_file_grep']
    response_files = [e['content'] for e in trace_events if e['type'] == 'found_file_response']
    entities_found = [e['content'] for e in trace_events if e['type'] == 'found_entity']
    
    num_reasoning = len([e for e in trace_events if e['type'] == 'reasoning'])
    num_commands = len([e for e in trace_events if e['type'] == 'command'])
    
    return {
        'trace_events': trace_events,  # Sequential trace
        'grep_files': grep_files,
        'response_files': response_files,
        'entities_found': entities_found,
        'num_reasoning_steps': num_reasoning,
        'num_commands': num_commands,
    }


def calculate_instance_metrics(prediction: Dict, ground_truth: Dict) -> Dict:
    """Calculate metrics for a single instance."""
    pred_files = set(prediction.get('found_files', []))
    pred_modules = set(prediction.get('found_modules', []))
    pred_entities = set(prediction.get('found_entities', []))
    
    # Extract ground truth
    gt_functions = ground_truth.get('edit_functions', [])
    
    # File level
    gt_files = set()
    for func in gt_functions:
        file_path = func.split(':')[0]
        gt_files.add(file_path)
    
    # Module level
    gt_modules = set()
    for func in gt_functions:
        parts = func.split(':')
        if len(parts) >= 2:
            fn = parts[0]
            func_part = parts[1]
            if '.' in func_part:
                class_name = func_part.split('.')[0]
                gt_modules.add(f'{fn}:{class_name}')
            else:
                gt_modules.add(func)
    
    # Function level
    gt_entities = set(gt_functions)
    
    # Calculate metrics
    def calc_precision_recall_f1(pred: set, gt: set) -> Dict:
        if not pred and not gt:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gt) if gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    file_metrics = calc_precision_recall_f1(pred_files, gt_files)
    module_metrics = calc_precision_recall_f1(pred_modules, gt_modules)
    entity_metrics = calc_precision_recall_f1(pred_entities, gt_entities)
    
    # Top-k accuracy
    def top_k_hit(pred_list: List, gt_set: set, k: int) -> bool:
        return any(p in gt_set for p in pred_list[:k])
    
    pred_files_list = prediction.get('found_files', [])
    pred_entities_list = prediction.get('found_entities', [])
    
    return {
        'file_level': file_metrics,
        'module_level': module_metrics,
        'entity_level': entity_metrics,
        'hit_at_1_file': top_k_hit(pred_files_list, gt_files, 1),
        'hit_at_3_file': top_k_hit(pred_files_list, gt_files, 3),
        'hit_at_5_file': top_k_hit(pred_files_list, gt_files, 5),
        'hit_at_1_entity': top_k_hit(pred_entities_list, gt_entities, 1),
        'hit_at_3_entity': top_k_hit(pred_entities_list, gt_entities, 3),
        'hit_at_5_entity': top_k_hit(pred_entities_list, gt_entities, 5),
        'num_pred_files': len(pred_files),
        'num_gt_files': len(gt_files),
        'num_pred_entities': len(pred_entities),
        'num_gt_entities': len(gt_entities),
    }


def categorize_failure(prediction: Dict, metrics: Dict, trace: Dict) -> str:
    """Categorize the type of failure - ONLY agent performance issues, not system issues."""
    status = prediction.get('status', 'UNKNOWN')
    
    # System failures - NOT categorized as agent failures
    if status in ['TIMEOUT', 'RATE_LIMITED', 'FAILED']:
        return 'system_error'  # Will be filtered out
    
    if status != 'FINISHED':
        return 'system_error'  # Will be filtered out
    
    # For FINISHED instances, categorize by agent performance
    file_recall = metrics['file_level']['recall']
    file_precision = metrics['file_level']['precision']
    entity_recall = metrics['entity_level']['recall']
    entity_precision = metrics['entity_level']['precision']
    
    # Perfect performance
    if file_recall == 1.0 and entity_recall == 1.0:
        return 'success'
    
    # Agent performance failures
    if file_recall == 0:
        return 'complete_miss'  # Found nothing
    elif file_recall < 0.3:
        return 'severe_recall_failure'  # Found very few correct files
    elif file_precision < 0.3:
        return 'severe_precision_failure'  # Mostly wrong files
    elif entity_recall == 0 and file_recall > 0:
        return 'file_only_no_entities'  # Found files but no functions
    elif entity_recall < 0.3:
        return 'poor_entity_localization'  # Very poor at function level
    elif file_recall < 0.7 or entity_recall < 0.5:
        return 'moderate_performance'  # Decent but not great
    else:
        return 'success'  # Good performance


def create_analysis_record(
    instance_id: str,
    prediction: Dict,
    ground_truth: Dict,
    trace: Dict
) -> Dict:
    """Create comprehensive analysis record for one instance."""
    metrics = calculate_instance_metrics(prediction, ground_truth)
    failure_category = categorize_failure(prediction, metrics, trace)
    
    # Determine success - only agent performance matters
    # System errors are not counted as agent failures
    is_success = (
        failure_category == 'success' or
        (prediction.get('status') == 'FINISHED' and
         metrics['file_level']['recall'] >= 0.7 and
         metrics['entity_level']['recall'] >= 0.5)
    )
    
    # System failure vs agent failure
    is_system_failure = failure_category == 'system_error'
    
    # Get ground truth info
    gt_files = set()
    gt_entities = set()
    for func in ground_truth.get('edit_functions', []):
        file_path = func.split(':')[0]
        gt_files.add(file_path)
        gt_entities.add(func)
    
    return {
        'instance_id': instance_id,
        'repo': ground_truth.get('repo', 'unknown'),
        'problem_statement': ground_truth.get('problem_statement', '')[:500],  # Truncate
        
        # Status
        'status': prediction.get('status', 'UNKNOWN'),
        'is_success': is_success,
        'is_system_failure': is_system_failure,  # New flag
        'failure_category': failure_category,
        
        # Predictions
        'predicted_files': prediction.get('found_files', []),
        'predicted_modules': prediction.get('found_modules', []),
        'predicted_entities': prediction.get('found_entities', []),
        
        # Ground truth
        'ground_truth_files': sorted(list(gt_files)),
        'ground_truth_entities': sorted(list(gt_entities)),
        
        # Metrics
        'metrics': metrics,
        
        # Execution trace (sequential)
        'trace': trace,
        
        # Diagnosis
        'diagnosis': generate_diagnosis(prediction, metrics, trace, gt_files, gt_entities)
    }


def generate_diagnosis(
    prediction: Dict,
    metrics: Dict,
    trace: Dict,
    gt_files: set,
    gt_entities: set
) -> Dict:
    """Generate diagnostic information about what went wrong/right."""
    pred_files = set(prediction.get('found_files', []))
    pred_entities = set(prediction.get('found_entities', []))
    
    return {
        'correct_files': sorted(list(pred_files & gt_files)),
        'missed_files': sorted(list(gt_files - pred_files)),
        'incorrect_files': sorted(list(pred_files - gt_files)),
        
        'correct_entities': sorted(list(pred_entities & gt_entities)),
        'missed_entities': sorted(list(gt_entities - pred_entities)),
        'incorrect_entities': sorted(list(pred_entities - gt_entities)),
        
        'grep_found_correct_files': sorted(list(set(trace.get('grep_files', [])) & gt_files)),
        'response_found_correct_files': sorted(list(set(trace.get('response_files', [])) & gt_files)),
        
        'used_fallback': len(trace.get('grep_files', [])) == 0 and len(trace.get('response_files', [])) == 0 and len(pred_files) > 0,
    }


def create_markdown_report(analysis_records: List[Dict], output_path: str):
    """Create human/LLM readable markdown analysis report."""
    
    # Filter out system failures - only analyze agent performance
    agent_performance_records = [r for r in analysis_records if not r.get('is_system_failure', False)]
    system_failure_records = [r for r in analysis_records if r.get('is_system_failure', False)]
    
    # Categorize by agent performance
    success_records = [r for r in agent_performance_records if r['is_success']]
    failure_records = [r for r in agent_performance_records if not r['is_success']]
    
    # Failure categorization
    failure_categories = Counter(r['failure_category'] for r in failure_records)
    
    with open(output_path, 'w') as f:
        f.write("# Codex Agent Execution Analysis Report\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Instances Analyzed**: {len(agent_performance_records)}\n")
        f.write(f"- **System Failures (Excluded)**: {len(system_failure_records)} (timeout, rate limit, git errors)\n")
        f.write(f"- **Agent Successes**: {len(success_records)} ({100*len(success_records)/len(agent_performance_records):.1f}%)\n")
        f.write(f"- **Agent Failures**: {len(failure_records)} ({100*len(failure_records)/len(agent_performance_records):.1f}%)\n\n")
        
        # Failure breakdown
        if failure_categories:
            f.write("### Agent Failure Breakdown\n\n")
            for category, count in failure_categories.most_common():
                if category != 'system_error':
                    f.write(f"- **{category.replace('_', ' ').title()}**: {count} instances ({100*count/len(failure_records):.1f}% of failures)\n")
            f.write("\n")
        
        # Average metrics (only for agent performance records)
        f.write("## Average Metrics (Agent Performance Only)\n\n")
        if agent_performance_records:
            avg_file_recall = sum(r['metrics']['file_level']['recall'] for r in agent_performance_records) / len(agent_performance_records)
            avg_file_precision = sum(r['metrics']['file_level']['precision'] for r in agent_performance_records) / len(agent_performance_records)
            avg_entity_recall = sum(r['metrics']['entity_level']['recall'] for r in agent_performance_records) / len(agent_performance_records)
            avg_entity_precision = sum(r['metrics']['entity_level']['precision'] for r in agent_performance_records) / len(agent_performance_records)
            
            f.write(f"- **File-level Recall**: {avg_file_recall:.3f}\n")
            f.write(f"- **File-level Precision**: {avg_file_precision:.3f}\n")
            f.write(f"- **Entity-level Recall**: {avg_entity_recall:.3f}\n")
            f.write(f"- **Entity-level Precision**: {avg_entity_precision:.3f}\n\n")
        
        # Execution patterns
        f.write("## Execution Patterns\n\n")
        avg_reasoning = sum(r['trace'].get('num_reasoning_steps', 0) for r in agent_performance_records) / len(agent_performance_records)
        avg_commands = sum(r['trace'].get('num_commands', 0) for r in agent_performance_records) / len(agent_performance_records)
        
        f.write(f"- **Avg Reasoning Steps**: {avg_reasoning:.1f}\n")
        f.write(f"- **Avg Commands Executed**: {avg_commands:.1f}\n\n")
        
        # Most common commands
        all_commands = []
        for r in agent_performance_records:
            trace_events = r['trace'].get('trace_events', [])
            commands = [e['content'] for e in trace_events if e['type'] == 'command']
            all_commands.extend(commands)
        
        # Extract command names (first tool)
        command_types = []
        for cmd in all_commands:
            # Extract tool name: rg, sed, find, etc.
            parts = cmd.split()
            if parts:
                # Remove /usr/bin/bash -lc wrapper
                if 'bash' in parts[0]:
                    if len(parts) > 2:
                        tool = parts[2].strip('"\'')
                        command_types.append(tool.split()[0])
                else:
                    command_types.append(parts[0].split('/')[-1])
        
        command_counter = Counter(command_types)
        f.write("### Most Common Tools Used\n\n")
        for cmd, count in command_counter.most_common(10):
            f.write(f"- `{cmd}`: {count} times\n")
        f.write("\n")
        
        # Sample failure cases with SEQUENTIAL TRACES
        f.write("## Sample Failure Cases (Agent Performance Issues)\n\n")
        for category, count in failure_categories.most_common(3):
            if category == 'system_error':
                continue
                
            category_failures = [r for r in failure_records if r['failure_category'] == category][:2]
            
            f.write(f"### {category.replace('_', ' ').title()} ({count} instances)\n\n")
            
            for record in category_failures:
                f.write(f"#### Instance: `{record['instance_id']}`\n\n")
                f.write(f"**Problem**: {record['problem_statement'][:200]}...\n\n")
                
                f.write(f"**Ground Truth Files**: {', '.join(record['ground_truth_files'][:5])}\n\n")
                f.write(f"**Predicted Files**: {', '.join(record['predicted_files'][:5]) if record['predicted_files'] else '(none)'}\n\n")
                
                diag = record['diagnosis']
                f.write(f"**Performance**:\n")
                f.write(f"- File Recall: {record['metrics']['file_level']['recall']:.3f}\n")
                f.write(f"- File Precision: {record['metrics']['file_level']['precision']:.3f}\n")
                f.write(f"- Entity Recall: {record['metrics']['entity_level']['recall']:.3f}\n")
                f.write(f"- Correct: {len(diag['correct_files'])} files, {len(diag['correct_entities'])} entities\n")
                f.write(f"- Missed: {len(diag['missed_files'])} files, {len(diag['missed_entities'])} entities\n\n")
                
                if diag['missed_files']:
                    f.write(f"**Missed Files**: {', '.join(diag['missed_files'][:3])}\n\n")
                
                # Show SEQUENTIAL execution trace
                trace_events = record['trace'].get('trace_events', [])
                if trace_events:
                    f.write(f"**Execution Trace** ({len(trace_events)} events):\n\n")
                    for i, event in enumerate(trace_events[:15], 1):  # Show first 15 events
                        event_type = event['type']
                        content = event['content'][:150]
                        
                        if event_type == 'reasoning':
                            f.write(f"{i}. **[Reasoning]** {content}...\n\n")
                        elif event_type == 'command':
                            f.write(f"{i}. **[Command]** `{content}`\n\n")
                        elif event_type == 'output':
                            f.write(f"{i}. **[Output]** `{content}`\n\n")
                        elif event_type == 'found_file_grep':
                            f.write(f"{i}. **[Found File]** {content}\n\n")
                        elif event_type == 'assistant_response':
                            f.write(f"{i}. **[Final Response]** {content}...\n\n")
                    
                    if len(trace_events) > 15:
                        f.write(f"... ({len(trace_events) - 15} more events)\n\n")
                
                f.write("---\n\n")
        
        # Sample success cases
        f.write("## Sample Success Cases\n\n")
        for record in success_records[:3]:
            f.write(f"### Instance: `{record['instance_id']}`\n\n")
            f.write(f"**Problem**: {record['problem_statement'][:200]}...\n\n")
            
            f.write(f"**Metrics**:\n")
            f.write(f"- File Recall: {record['metrics']['file_level']['recall']:.3f}\n")
            f.write(f"- Entity Recall: {record['metrics']['entity_level']['recall']:.3f}\n\n")
            
            # Show what was found correctly
            diag = record['diagnosis']
            if diag['correct_files']:
                f.write(f"**Correctly Located Files**: {', '.join(diag['correct_files'])}\n\n")
            
            # Show abbreviated trace for success
            trace_events = record['trace'].get('trace_events', [])
            f.write(f"**Execution**: {record['trace'].get('num_reasoning_steps', 0)} reasoning steps, {record['trace'].get('num_commands', 0)} commands\n\n")
            
            f.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Codex agent data for LLM analysis")
    parser.add_argument("--loc_file", type=str, required=True,
                        help="Path to loc_outputs.jsonl (predictions)")
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Path to ground truth file (mulocbench.jsonl)")
    parser.add_argument("--log_file", type=str, required=True,
                        help="Path to log file (supports glob patterns)")
    parser.add_argument("--output_dir", type=str, default="analysis/",
                        help="Output directory for analysis files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    
    # Load predictions
    predictions = load_jsonl(args.loc_file)
    pred_dict = {p['instance_id']: p for p in predictions}
    
    # Load ground truth
    ground_truth = load_jsonl(args.gt_file)
    gt_dict = {g['instance_id']: g for g in ground_truth}
    
    # Load log file(s)
    log_files = glob.glob(args.log_file)
    if not log_files:
        print(f"Warning: No log files found matching: {args.log_file}")
        log_content = ""
    else:
        print(f"Loading {len(log_files)} log file(s)...")
        log_content = ""
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_content += f.read() + "\n"
    
    print(f"Processing {len(predictions)} instances...")
    
    # Create analysis records
    analysis_records = []
    for pred in predictions:
        instance_id = pred['instance_id']
        
        # Get ground truth
        if instance_id not in gt_dict:
            print(f"Warning: No ground truth for {instance_id}")
            continue
        
        gt = gt_dict[instance_id]
        
        # Parse trace from log
        trace = parse_log_for_instance(log_content, instance_id) if log_content else {}
        
        # Create analysis record
        record = create_analysis_record(instance_id, pred, gt, trace)
        analysis_records.append(record)
    
    print(f"Created {len(analysis_records)} analysis records")
    
    # Separate success and failure cases (exclude system failures)
    success_records = [r for r in analysis_records if r['is_success']]
    failure_records = [r for r in analysis_records if not r['is_success'] and not r.get('is_system_failure', False)]
    system_failures = [r for r in analysis_records if r.get('is_system_failure', False)]
    
    print(f"  Success: {len(success_records)}")
    print(f"  Agent Failures: {len(failure_records)} (performance issues)")
    print(f"  System Failures: {len(system_failures)} (timeout/rate-limit/git errors - excluded from analysis)")
    
    # Save full analysis data
    full_output = output_dir / "analysis_data.jsonl"
    with open(full_output, 'w') as f:
        for record in analysis_records:
            f.write(json.dumps(record) + '\n')
    print(f"\nSaved full analysis: {full_output}")
    
    # Save failure cases
    failure_output = output_dir / "failure_cases.jsonl"
    with open(failure_output, 'w') as f:
        for record in failure_records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved failure cases: {failure_output}")
    
    # Save success cases
    success_output = output_dir / "success_cases.jsonl"
    with open(success_output, 'w') as f:
        for record in success_records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved success cases: {success_output}")
    
    # Create markdown report
    report_output = output_dir / "analysis_report.md"
    create_markdown_report(analysis_records, report_output)
    print(f"Saved markdown report: {report_output}")
    
    # Create summary CSV
    summary_data = []
    for record in analysis_records:
        summary_data.append({
            'instance_id': record['instance_id'],
            'repo': record['repo'],
            'status': record['status'],
            'is_success': record['is_success'],
            'is_system_failure': record.get('is_system_failure', False),
            'failure_category': record['failure_category'],
            'file_recall': record['metrics']['file_level']['recall'],
            'file_precision': record['metrics']['file_level']['precision'],
            'entity_recall': record['metrics']['entity_level']['recall'],
            'entity_precision': record['metrics']['entity_level']['precision'],
            'num_reasoning_steps': record['trace'].get('num_reasoning_steps', 0),
            'num_commands': record['trace'].get('num_commands', 0),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")
    
    print("\n" + "="*60)
    print("Analysis preparation complete!")
    print("="*60)
    print("\nFiles created:")
    print(f"  1. {full_output} - Complete structured data")
    print(f"  2. {failure_output} - Failure cases only")
    print(f"  3. {success_output} - Success cases only")
    print(f"  4. {report_output} - Human/LLM readable report")
    print(f"  5. {summary_csv} - Summary statistics")
    print("\nRecommended LLM analysis prompts:")
    print("  - Upload analysis_report.md for high-level insights")
    print("  - Upload failure_cases.jsonl for deep failure analysis")
    print("  - Compare success_cases.jsonl vs failure_cases.jsonl for patterns")


if __name__ == "__main__":
    main()