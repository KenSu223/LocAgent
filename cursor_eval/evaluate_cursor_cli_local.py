"""
MULocBench Evaluation Script for Cursor CLI (Local Agent)

Evaluates Cursor's local CLI agent on code localization tasks.
This script clones repos, checks out specific commits, and runs
cursor-agent locally in non-interactive mode.

Prerequisites:
    1. Install Cursor CLI: curl https://cursor.com/install -fsSL | bash
    2. Authenticate: cursor-agent login (or set CURSOR_API_KEY)
    3. Git must be available

Usage:
    python evaluate_cursor_cli_local.py \
        --dataset_path data/mulocbench/mulocbench.jsonl \
        --output_dir results/cursor_cli \
        --repos scikit-learn flask requests transformers pandas \
        --num_samples 50 \
        --model claude-4-sonnet
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    model: str = "claude-4-sonnet"  # Cursor model to use
    timeout: int = 300  # Timeout per instance in seconds
    num_workers: int = 1  # Number of parallel workers (careful with rate limits)
    repo_cache_dir: str = "/tmp/mulocbench_repos"
    num_samples: Optional[int] = None
    force_write: bool = False  # Allow agent to write files (--force flag)
    include_tests: bool = False  # Include test files in ground truth


@dataclass 
class RepoManager:
    """Manages repository cloning and caching"""
    cache_dir: str
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_repo_path(self, repo: str, commit: str) -> str:
        """Get path to repo at specific commit, cloning if needed"""
        # Create a unique path for this repo+commit combination
        safe_name = repo.replace("/", "__")
        repo_path = os.path.join(self.cache_dir, f"{safe_name}__{commit[:12]}")
        
        with self._lock:
            if os.path.exists(repo_path):
                return repo_path
            
            # Clone the repo
            clone_url = f"https://github.com/{repo}.git"
            logger.info(f"Cloning {repo} at {commit[:12]}...")
            
            try:
                # Clone with minimal history for speed
                subprocess.run(
                    ["git", "clone", "--quiet", clone_url, repo_path],
                    check=True,
                    capture_output=True,
                    timeout=120
                )
                
                # Checkout specific commit
                subprocess.run(
                    ["git", "checkout", "--quiet", commit],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=30
                )
                
                logger.info(f"Successfully checked out {repo} at {commit[:12]}")
                return repo_path
                
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout cloning {repo}")
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)
                raise
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone {repo}: {e.stderr.decode() if e.stderr else str(e)}")
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)
                raise


def build_localization_prompt(instance: Dict[str, Any]) -> str:
    """Build the prompt for code localization task - aligned with LocAgent's methodology"""
    problem_statement = instance.get("problem_statement", "")
    repo = instance.get("repo", "")
    
    # Extract package name from repo (e.g., "pallets/flask" -> "flask")
    package_name = repo.split("/")[-1] if repo else "the repository"
    
    # System context (similar to LocAgent's SYSTEM_PROMPT)
    system_context = """You're an experienced software tester and static analysis expert. 
Given the problem offered by the user, please perform a thorough static analysis and to localize the bug in this repository using the available tools.
Analyze the execution flow of this code step by step, as if you were a human tester mentally running it.

Focus on:
- Tracing the flow of execution through critical paths, conditions, loops, and function calls.
- Identifying any deviations, potential errors, or unexpected behavior that could contribute to the issue.
- Considering how dynamic binding, late resolution, or other runtime behavior may influence the code's behavior.
- Highlighting possible root causes or key areas for further inspection."""

    # Task instruction (aligned with LocAgent's TASK_INSTRUCTION)
    task_instruction = f"""
Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.

Follow these steps to localize the issue:

## Step 1: Categorize and Extract Key Problem Information
 - Classify the problem statement into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Identify modules in the '{package_name}' package mentioned in each category.
 - Use extracted keywords and line numbers to search for relevant code references for additional context.

## Step 2: Locate Referenced Modules
- Accurately determine specific modules
    - Explore the repo to familiarize yourself with its structure.
    - Analyze the described execution flow to identify specific modules or components being referenced.
- Pay special attention to distinguishing between modules with similar names using context and described execution flow.
- Output Format for collected relevant modules:
    - Use the format: 'file_path:QualifiedName'
    - E.g., for a function `calculate_sum` in the `MathUtils` class located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.

## Step 3: Analyze and Reproducing the Problem
- Clarify the Purpose of the Issue
    - If expanding capabilities: Identify where and how to incorporate new behavior, fields, or modules.
    - If addressing unexpected behavior: Focus on localizing modules containing potential bugs.
- Reconstruct the execution flow
    - Identify main entry points triggering the issue.
    - Trace function calls, class interactions, and sequences of events.
    - Identify potential breakpoints causing the issue.
    Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

## Step 4: Locate Areas for Modification
- Locate specific files, functions, or lines of code requiring changes or containing critical information for resolving the issue.
- Consider upstream and downstream dependencies that may affect or be affected by the issue.
- If applicable, identify where to introduce new fields, functions, or variables.
- Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

## Output Format for Final Results:
After your analysis, provide your final answer in this exact format:

FILES:
- path/to/file1.py
- path/to/file2.py

FUNCTIONS:
- path/to/file1.py:function_name
- path/to/file1.py:ClassName.method_name
- path/to/file2.py:another_function

Your answer should include about 5 files. Be specific and precise. Only list locations that actually need modification.

Note: Your thinking should be thorough and so it's fine if it's very long.
"""

    prompt = f"""{system_context}

{task_instruction}

## GitHub Problem Description:
{problem_statement}
"""

    return prompt


def run_cursor_agent(
    repo_path: str,
    prompt: str,
    config: EvaluationConfig
) -> Tuple[str, bool]:
    """
    Run cursor-agent in the specified repo directory.
    
    Returns:
        Tuple of (output_text, success)
    """
    # Build the command
    cmd = [
        "cursor-agent",
        "-p",  # Print mode (non-interactive)
        "--output-format", "text",
        "--model", config.model,
    ]
    
    if config.force_write:
        cmd.append("--force")
    
    # Add the prompt
    cmd.append(prompt)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            env={**os.environ, "NO_COLOR": "1"}  # Disable color codes
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
        return output, success
        
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {config.timeout}s", False
    except FileNotFoundError:
        return "cursor-agent not found. Install with: curl https://cursor.com/install -fsSL | bash", False
    except Exception as e:
        return f"Error: {str(e)}", False


def parse_agent_response(response: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse agent response to extract files, modules, and entities.
    
    Returns:
        Tuple of (found_files, found_modules, found_entities)
        
    LocAgent format:
        - found_files: ["path/to/file.py", ...]
        - found_modules: ["path/to/file.py:ClassName", ...]  (class-level)
        - found_entities: ["path/to/file.py:ClassName.method", "path/to/file.py:function", ...]
    """
    found_files = []
    found_modules = []
    found_entities = []
    
    # Try to find FILES: section
    files_match = re.search(r'FILES:\s*\n((?:[-*]\s*.+\n?)+)', response, re.IGNORECASE)
    if files_match:
        files_section = files_match.group(1)
        for line in files_section.strip().split('\n'):
            line = line.strip()
            if line.startswith(('-', '*')):
                file_path = line.lstrip('-* ').strip()
                if file_path and file_path.endswith('.py'):
                    if file_path not in found_files:
                        found_files.append(file_path)
    
    # Try to find FUNCTIONS: section
    funcs_match = re.search(r'FUNCTIONS:\s*\n((?:[-*]\s*.+\n?)+)', response, re.IGNORECASE)
    if funcs_match:
        funcs_section = funcs_match.group(1)
        for line in funcs_section.strip().split('\n'):
            line = line.strip()
            if line.startswith(('-', '*')):
                func_loc = line.lstrip('-* ').strip()
                if func_loc and ':' in func_loc:
                    # Add to found_entities
                    if func_loc not in found_entities:
                        found_entities.append(func_loc)
                    
                    # Extract file
                    file_path = func_loc.split(':')[0]
                    if file_path.endswith('.py') and file_path not in found_files:
                        found_files.append(file_path)
                    
                    # Extract module (class-level)
                    # "file.py:Class.method" -> "file.py:Class"
                    # "file.py:function" -> "file.py:function" (top-level function counts as module)
                    parts = func_loc.split(':')
                    if len(parts) >= 2:
                        func_part = parts[1]
                        if '.' in func_part:
                            # Class.method -> extract Class
                            class_name = func_part.split('.')[0]
                            module_loc = f"{parts[0]}:{class_name}"
                        else:
                            # Top-level function
                            module_loc = func_loc
                        if module_loc not in found_modules:
                            found_modules.append(module_loc)
    
    # Fallback: extract patterns like "file.py:function" or "file.py:Class.method"
    if not found_entities:
        pattern = r'([a-zA-Z0-9_/]+\.py):([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        for match in re.finditer(pattern, response):
            entity = f"{match.group(1)}:{match.group(2)}"
            if entity not in found_entities:
                found_entities.append(entity)
                
                # Extract file
                file_path = match.group(1)
                if file_path not in found_files:
                    found_files.append(file_path)
                
                # Extract module
                func_part = match.group(2)
                if '.' in func_part:
                    class_name = func_part.split('.')[0]
                    module_loc = f"{file_path}:{class_name}"
                else:
                    module_loc = entity
                if module_loc not in found_modules:
                    found_modules.append(module_loc)
    
    return found_files, found_modules, found_entities


def evaluate_instance(
    instance: Dict[str, Any],
    repo_manager: RepoManager,
    config: EvaluationConfig
) -> Dict[str, Any]:
    """Evaluate a single instance and return LocAgent-compatible output"""
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo", "")
    commit = instance.get("base_commit", "")
    
    # LocAgent-compatible output format
    result = {
        "instance_id": instance_id,
        "found_files": [],
        "found_modules": [],
        "found_entities": [],
        "raw_output_loc": [],  # Store raw agent responses
        "meta_data": {
            "repo": repo,
            "base_commit": commit,
            "problem_statement": instance.get("problem_statement", ""),
            "status": "PENDING",
            "error": None
        }
    }
    
    if not commit:
        result["meta_data"]["status"] = "SKIPPED"
        result["meta_data"]["error"] = "No base_commit specified"
        return result
    
    try:
        # Get repo path (clone if needed)
        repo_path = repo_manager.get_repo_path(repo, commit)
        
        # Build prompt
        prompt = build_localization_prompt(instance)
        
        # Run cursor agent
        logger.info(f"Running cursor-agent for {instance_id}...")
        response, success = run_cursor_agent(repo_path, prompt, config)
        
        result["raw_output_loc"].append(response)
        
        if not success:
            result["meta_data"]["status"] = "FAILED"
            result["meta_data"]["error"] = response[:500]  # Truncate error
            return result
        
        # Parse response into LocAgent format
        found_files, found_modules, found_entities = parse_agent_response(response)
        
        result["found_files"] = found_files
        result["found_modules"] = found_modules
        result["found_entities"] = found_entities
        result["meta_data"]["status"] = "FINISHED"
        
        return result
        
    except Exception as e:
        result["meta_data"]["status"] = "ERROR"
        result["meta_data"]["error"] = str(e)
        return result


def compute_metrics(results: List[Dict[str, Any]], include_tests: bool = False) -> Dict[str, Any]:
    """Compute basic evaluation metrics (for quick feedback during evaluation)"""
    
    total = len(results)
    finished = [r for r in results if r.get("meta_data", {}).get("status") == "FINISHED"]
    failed = [r for r in results if r.get("meta_data", {}).get("status") == "FAILED"]
    errors = [r for r in results if r.get("meta_data", {}).get("status") == "ERROR"]
    skipped = [r for r in results if r.get("meta_data", {}).get("status") == "SKIPPED"]
    
    metrics = {
        "total_instances": total,
        "finished": len(finished),
        "failed": len(failed),
        "errors": len(errors),
        "skipped": len(skipped),
        "avg_files_found": sum(len(r.get("found_files", [])) for r in finished) / max(len(finished), 1),
        "avg_modules_found": sum(len(r.get("found_modules", [])) for r in finished) / max(len(finished), 1),
        "avg_entities_found": sum(len(r.get("found_entities", [])) for r in finished) / max(len(finished), 1),
    }
    
    return metrics


def is_valid_result(result: Dict[str, Any]) -> bool:
    """Check if a result has valid (non-empty) predictions"""
    has_files = len(result.get("found_files", [])) > 0
    has_entities = len(result.get("found_entities", [])) > 0
    status = result.get("meta_data", {}).get("status", "")
    return (has_files or has_entities) and status == "FINISHED"


def load_existing_results(output_path: str) -> Tuple[Dict[str, Dict], set, set]:
    """
    Load existing results from output file.
    
    Returns:
        Tuple of (all_results_dict, completed_ids, empty_ids)
    """
    existing_results = {}
    completed_ids = set()
    empty_ids = set()
    
    if not os.path.exists(output_path):
        return existing_results, completed_ids, empty_ids
    
    logger.info(f"Loading existing results from {output_path}...")
    
    with open(output_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    instance_id = result.get("instance_id")
                    if instance_id:
                        existing_results[instance_id] = result
                        if is_valid_result(result):
                            completed_ids.add(instance_id)
                        else:
                            empty_ids.add(instance_id)
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"  Completed (with results): {len(completed_ids)}")
    logger.info(f"  Empty (need retry): {len(empty_ids)}")
    
    return existing_results, completed_ids, empty_ids


def write_valid_results_to_file(
    output_path: str,
    existing_results: Dict[str, Dict],
    completed_ids: set
) -> None:
    """
    Write only valid/completed results to file (overwrite mode).
    This sets up the file for incremental appending of new results.
    """
    with open(output_path, 'w') as f:
        for instance_id in completed_ids:
            if instance_id in existing_results:
                f.write(json.dumps(existing_results[instance_id]) + '\n')


def append_result_to_file(output_path: str, result: Dict[str, Any], file_lock: Lock) -> None:
    """Append a single result to the output file (thread-safe)"""
    with file_lock:
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')


def run_evaluation(
    dataset_path: str,
    output_dir: str,
    config: EvaluationConfig,
    repos_filter: Optional[List[str]] = None
):
    """Run full evaluation with incremental writing and resume capability"""
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    instances = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    # Filter by repository if specified
    if repos_filter:
        filter_terms = [r.lower().replace("-", "").replace("_", "") for r in repos_filter]
        
        def matches_filter(repo: str) -> bool:
            repo_name = repo.split("/")[-1].lower().replace("-", "").replace("_", "")
            return any(term in repo_name or repo_name in term for term in filter_terms)
        
        original_count = len(instances)
        instances = [inst for inst in instances if matches_filter(inst.get("repo", ""))]
        logger.info(f"Filtered to {len(instances)}/{original_count} instances matching repos: {repos_filter}")
        
        # Show matched repos
        matched_repos = sorted(set(inst.get("repo", "") for inst in instances))
        for repo in matched_repos:
            count = sum(1 for i in instances if i.get("repo") == repo)
            logger.info(f"  {repo}: {count} instances")
    
    if config.num_samples:
        instances = instances[:config.num_samples]
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    repo_manager = RepoManager(cache_dir=config.repo_cache_dir)
    output_path = os.path.join(output_dir, "loc_outputs.jsonl")
    
    # Load existing results
    existing_results, completed_ids, empty_ids = load_existing_results(output_path)
    
    # Determine which instances need to be run:
    # 1. Not present in output file at all (new instances, e.g., from added repos)
    # 2. Present but empty/failed (need retry)
    instances_to_run = []
    instance_ids_in_dataset = set()
    
    for inst in instances:
        instance_id = inst.get("instance_id")
        instance_ids_in_dataset.add(instance_id)
        
        if instance_id in completed_ids:
            continue  # Skip - already has valid results
        
        # Either not in file or empty/failed
        instances_to_run.append(inst)
    
    # Separate counts for logging
    new_instances = [inst for inst in instances_to_run if inst.get("instance_id") not in existing_results]
    retry_instances = [inst for inst in instances_to_run if inst.get("instance_id") in empty_ids]
    
    logger.info(f"Instances to process:")
    logger.info(f"  New (not in output file): {len(new_instances)}")
    logger.info(f"  Retry (empty/failed): {len(retry_instances)}")
    logger.info(f"  Total to run: {len(instances_to_run)}")
    logger.info(f"  Skipping (already completed): {len(completed_ids & instance_ids_in_dataset)}")
    
    if len(instances_to_run) == 0:
        logger.info("All instances already completed!")
        # Compute and show metrics from existing results
        all_results = [existing_results[iid] for iid in instance_ids_in_dataset if iid in existing_results]
        metrics = compute_metrics(all_results, config.include_tests)
        _print_final_results(metrics, output_path)
        return all_results, metrics
    
    # Write valid/completed results to file first (overwrite mode)
    # Then we'll append new results incrementally
    logger.info(f"Writing {len(completed_ids)} valid existing results to output file...")
    write_valid_results_to_file(output_path, existing_results, completed_ids)
    
    # Create a lock for thread-safe file writing
    file_lock = Lock()
    
    # Run evaluation with incremental writing
    logger.info(f"Evaluating {len(instances_to_run)} instances (results written incrementally)...")
    
    if config.num_workers == 1:
        # Sequential execution with incremental writes
        for i, instance in enumerate(tqdm(instances_to_run, desc="Evaluating")):
            result = evaluate_instance(instance, repo_manager, config)
            
            # Write immediately to file
            append_result_to_file(output_path, result, file_lock)
            
            # Update existing_results for final metrics
            existing_results[result.get("instance_id")] = result
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(instances_to_run)} - Last: {result.get('instance_id')}")
    else:
        # Parallel execution with incremental writes
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(evaluate_instance, inst, repo_manager, config): inst
                for inst in instances_to_run
            }
            
            for future in tqdm(as_completed(futures), total=len(instances_to_run), desc="Evaluating"):
                result = future.result()
                
                # Write immediately to file (thread-safe)
                append_result_to_file(output_path, result, file_lock)
                
                # Update existing_results for final metrics
                existing_results[result.get("instance_id")] = result
    
    # Compute final metrics from all results relevant to current filter
    final_results = [existing_results[iid] for iid in instance_ids_in_dataset if iid in existing_results]
    metrics = compute_metrics(final_results, config.include_tests)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    _print_final_results(metrics, output_path)
    
    return final_results, metrics


def _print_final_results(metrics: Dict[str, Any], output_path: str):
    """Print final evaluation results"""
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total instances: {metrics['total_instances']}")
    logger.info(f"  Finished: {metrics['finished']}")
    logger.info(f"  Failed: {metrics['failed']}")
    logger.info(f"  Errors: {metrics['errors']}")
    logger.info(f"  Skipped: {metrics['skipped']}")
    logger.info("")
    logger.info("AVERAGE PREDICTIONS PER INSTANCE:")
    logger.info(f"  Files found:    {metrics['avg_files_found']:.2f}")
    logger.info(f"  Modules found:  {metrics['avg_modules_found']:.2f}")
    logger.info(f"  Entities found: {metrics['avg_entities_found']:.2f}")
    logger.info("")
    logger.info(f"Output saved to: {output_path}")
    logger.info("")
    logger.info("To evaluate with LocAgent metrics, run:")
    logger.info(f"  python run_locagent_eval.py --loc_file {output_path} --only_predicted")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cursor CLI agent on MULocBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python evaluate_cursor_cli_local.py \\
        --dataset_path data/mulocbench/mulocbench.jsonl \\
        --output_dir results/cursor_cli

    # With specific repos and model
    python evaluate_cursor_cli_local.py \\
        --dataset_path data/mulocbench/mulocbench.jsonl \\
        --output_dir results/cursor_cli \\
        --repos scikit-learn flask requests \\
        --model claude-4-sonnet \\
        --num_samples 50
        
Prerequisites:
    1. Install Cursor CLI: curl https://cursor.com/install -fsSL | bash
    2. Authenticate with: cursor-agent login
       Or set CURSOR_API_KEY environment variable
"""
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to MULocBench JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific repos (e.g., --repos scikit-learn flask)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-4-sonnet",
        help="Cursor model to use (default: claude-4-sonnet)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, be careful with rate limits)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per instance in seconds (default: 300)"
    )
    parser.add_argument(
        "--repo_cache_dir",
        type=str,
        default="/tmp/mulocbench_repos",
        help="Directory to cache cloned repositories"
    )
    parser.add_argument(
        "--include_tests",
        action="store_true",
        help="Include test files in ground truth evaluation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow cursor-agent to write files (not recommended for localization)"
    )
    
    args = parser.parse_args()
    
    # Check if cursor-agent is available
    try:
        result = subprocess.run(
            ["cursor-agent", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        logger.info(f"Cursor CLI version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("cursor-agent not found!")
        logger.error("Install with: curl https://cursor.com/install -fsSL | bash")
        logger.error("Then authenticate with: cursor-agent login")
        return
    except Exception as e:
        logger.warning(f"Could not check cursor-agent version: {e}")
    
    config = EvaluationConfig(
        model=args.model,
        timeout=args.timeout,
        num_workers=args.num_workers,
        repo_cache_dir=args.repo_cache_dir,
        num_samples=args.num_samples,
        force_write=args.force,
        include_tests=args.include_tests
    )
    
    run_evaluation(args.dataset_path, args.output_dir, config, args.repos)


if __name__ == "__main__":
    main()
