#!/usr/bin/env python3
"""
Preparation steps:
export AZURE_OPENAI_API_KEY="your-azure-api-key"

Evaluate OpenAI Codex CLI agent on MULocBench/LocBench.
Uses `codex exec` non-interactive mode with verbose logging.

configuration at: ~/.codex/config.toml

python codex/codex_run.py \
    --dataset_path /home/tsu25/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --output_dir results/codex_agent \
    --repos_dir /home/tsu25/LocAgent/mulocbench_repos \
    --repos flask \
    --num_samples 1 \
    --model gpt-5.2 \
    --timeout 300 \
    --max_retries 3 \
    --retry_delay 30 \
    -v
"""

#!/usr/bin/env python3
"""
Evaluate OpenAI Codex CLI agent on MULocBench/LocBench.
Uses `codex exec` non-interactive mode with verbose logging.
"""

import json
import subprocess
import os
import re
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import argparse


def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("codex_eval")
    logger.setLevel(logging.DEBUG)
    
    # File handler - always verbose
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def clone_repo_at_commit(repo: str, commit: str, target_dir: str, logger: logging.Logger) -> bool:
    """Clone repo and checkout specific commit."""
    try:
        if not os.path.exists(target_dir):
            logger.info(f"  Cloning {repo}...")
            subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", target_dir],
                check=True, capture_output=True, timeout=120
            )
        logger.info(f"  Checking out {commit[:8]}...")
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=target_dir, check=True, capture_output=True
        )
        return True
    except Exception as e:
        logger.error(f"  Failed to setup repo: {e}")
        return False


def build_prompt(problem_statement: str) -> str:
    """Build the prompt for code localization."""
    return f"""You are a code localization expert. Given a bug report or feature request, identify the files and functions that need to be modified.

## Bug Report / Issue:
{problem_statement}

## Task:
Analyze this issue and identify:
1. Which files need to be modified
2. Which specific functions or methods need changes

## Output Format:
Provide your answer in this exact format:

FILES:
- path/to/file1.py
- path/to/file2.py

FUNCTIONS:
- path/to/file1.py:function_name
- path/to/file1.py:ClassName.method_name

Your answer should include about 5 files. Be specific and precise."""


def parse_structured_response(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse structured FILES/FUNCTIONS format from assistant text."""
    found_files = []
    found_modules = []
    found_entities = []
    
    lines = text.split('\n')
    in_files = False
    in_functions = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('FILES:'):
            in_files, in_functions = True, False
            continue
        if line.upper().startswith('FUNCTIONS:'):
            in_files, in_functions = False, True
            continue
        
        if line.startswith('- ') or line.startswith('* '):
            line = line[2:].strip()
        line = line.strip('`')
        
        if in_files and line and (line.endswith('.py') or '/' in line):
            if line not in found_files:
                found_files.append(line)
        
        if in_functions and line and ':' in line:
            parts = line.split(':', 1)
            file_path, func_part = parts[0], parts[1]
            entity = f"{file_path}:{func_part}"
            
            if entity not in found_entities:
                found_entities.append(entity)
            if file_path not in found_files:
                found_files.append(file_path)
            
            # Extract module
            if '.' in func_part:
                class_name = func_part.split('.')[0]
                module_loc = f"{file_path}:{class_name}"
            else:
                module_loc = entity
            if module_loc not in found_modules:
                found_modules.append(module_loc)
    
    return found_files, found_modules, found_entities


def parse_codex_response(response: str, logger: logging.Logger) -> Tuple[List[str], List[str], List[str]]:
    """Parse Codex response to extract files, modules, and entities."""
    found_files = []
    found_modules = []
    found_entities = []
    
    # Try to parse JSONL events
    for line in response.strip().split('\n'):
        try:
            event = json.loads(line)
            event_type = event.get('type', '')
            
            # Log reasoning steps
            if event_type == 'item.completed':
                item = event.get('item', {})
                item_type = item.get('type', item.get('item_type', ''))
                
                if item_type == 'reasoning':
                    reasoning_text = item.get('text', '')
                    logger.debug(f"    [REASONING] {reasoning_text[:200]}...")
                
                elif item_type == 'command_execution':
                    cmd = item.get('command', '')
                    output = item.get('aggregated_output', '')
                    exit_code = item.get('exit_code', '')
                    logger.debug(f"    [COMMAND] {cmd}")
                    logger.debug(f"    [OUTPUT] {output[:500]}..." if len(output) > 500 else f"    [OUTPUT] {output}")
                    
                    # Parse file paths from ripgrep/grep output (file.py:line: ...)
                    for match in re.finditer(r'^([a-zA-Z0-9_/\-]+\.py):\d+:', output, re.MULTILINE):
                        fpath = match.group(1)
                        if fpath not in found_files:
                            found_files.append(fpath)
                            logger.debug(f"    [FOUND FILE from grep] {fpath}")
                
                elif item_type == 'assistant_message':
                    text = item.get('text', '')
                    logger.debug(f"    [ASSISTANT] {text[:500]}...")
                    
                    # Parse structured output from assistant
                    parsed_files, parsed_modules, parsed_entities = parse_structured_response(text)
                    for f in parsed_files:
                        if f not in found_files:
                            found_files.append(f)
                            logger.debug(f"    [FOUND FILE from response] {f}")
                    for m in parsed_modules:
                        if m not in found_modules:
                            found_modules.append(m)
                    for e in parsed_entities:
                        if e not in found_entities:
                            found_entities.append(e)
                            logger.debug(f"    [FOUND ENTITY from response] {e}")
            
            elif event_type == 'error':
                logger.warning(f"    [ERROR] {event.get('message', '')}")
            
            elif event_type == 'turn.failed':
                error_msg = event.get('error', {}).get('message', '')
                logger.warning(f"    [TURN FAILED] {error_msg}")
                    
        except json.JSONDecodeError:
            continue
    
    # Fallback: regex search for Python files in entire response
    if not found_files:
        logger.debug("    [FALLBACK] Using regex to find files...")
        for match in re.finditer(r'([a-zA-Z0-9_/\-]+\.py)', response):
            fpath = match.group(1)
            # Filter out common false positives
            if fpath not in found_files and not fpath.startswith('python') and '/' in fpath:
                found_files.append(fpath)
                logger.debug(f"    [FOUND FILE from fallback] {fpath}")
    
    # Generate modules from found files if empty
    if not found_modules:
        for f in found_files:
            if f not in found_modules:
                found_modules.append(f)
    
    return found_files, found_modules, found_entities


def is_rate_limit_error(response: str) -> bool:
    """Check if response contains rate limit error."""
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded",
        "tokens per minute",
        "requests per minute",
        "TPM",
        "RPM"
    ]
    response_lower = response.lower()
    return any(indicator.lower() in response_lower for indicator in rate_limit_indicators)


def is_retriable_error(response: str) -> bool:
    """Check if error is retriable (rate limit, timeout, temporary failure)."""
    retriable_indicators = [
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "connection",
        "disconnected",
        "stream disconnected",
        "temporary",
        "retry"
    ]
    response_lower = response.lower()
    return any(indicator.lower() in response_lower for indicator in retriable_indicators)


def run_codex_agent(
    instance: dict, 
    repos_dir: str, 
    model: str,
    timeout: int,
    logger: logging.Logger,
    max_retries: int = 3,
    retry_delay: int = 60
) -> dict:
    """Run Codex CLI agent on a single instance with retry logic."""
    import time
    
    instance_id = instance['instance_id']
    repo = instance['repo']
    commit = instance['base_commit']
    problem = instance['problem_statement']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Instance: {instance_id}")
    logger.info(f"Repo: {repo} @ {commit[:8]}")
    logger.info(f"{'='*60}")
    
    result = {
        "instance_id": instance_id,
        "found_files": [],
        "found_modules": [],
        "found_entities": [],
        "status": "FAILED",
        "error": None,
        "raw_response": "",
        "retries": 0
    }
    
    # Setup repo
    repo_dir = os.path.join(repos_dir, repo.replace('/', '_'))
    if not clone_repo_at_commit(repo, commit, repo_dir, logger):
        result["error"] = "Failed to clone repo"
        return result
    
    # Build prompt
    prompt = build_prompt(problem)
    logger.debug(f"  Problem: {problem[:200]}...")
    
    # Build command
    cmd = [
        "codex", "exec", 
        "--json",
        "--sandbox", "danger-full-access",
        "--model", model,
        prompt
    ]
    
    # Retry loop
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"  Retry {attempt}/{max_retries} after {retry_delay}s delay...")
            time.sleep(retry_delay)
            # Exponential backoff: double the delay for next retry
            retry_delay = min(retry_delay * 2, 300)  # Cap at 5 minutes
        
        logger.info(f"  Running Codex agent (attempt {attempt + 1}/{max_retries + 1})...")
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            response = proc.stdout
            stderr = proc.stderr or ""
            combined_output = response + stderr
            
            result["raw_response"] = response
            result["retries"] = attempt
            
            # Log stderr if any
            if stderr:
                logger.debug(f"  [STDERR] {stderr[:500]}")
            
            # Check for rate limit errors
            if is_rate_limit_error(combined_output):
                logger.warning(f"  Rate limit hit!")
                if attempt < max_retries:
                    logger.info(f"  Will retry after delay...")
                    continue
                else:
                    result["error"] = "Rate limit exceeded (max retries reached)"
                    result["status"] = "RATE_LIMITED"
                    return result
            
            # Check for other retriable errors (like stream disconnection)
            if is_retriable_error(combined_output) and not response.strip():
                logger.warning(f"  Retriable error detected")
                if attempt < max_retries:
                    continue
                else:
                    result["error"] = "Retriable error (max retries reached)"
                    return result
            
            # Parse the response
            logger.info(f"  Parsing response...")
            found_files, found_modules, found_entities = parse_codex_response(response, logger)
            
            result["found_files"] = found_files
            result["found_modules"] = found_modules
            result["found_entities"] = found_entities
            result["status"] = "FINISHED"
            
            logger.info(f"  Results: {len(found_files)} files, {len(found_modules)} modules, {len(found_entities)} entities")
            if found_files:
                logger.info(f"  Files: {found_files[:5]}{'...' if len(found_files) > 5 else ''}")
            
            # Success - break out of retry loop
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"  TIMEOUT after {timeout}s")
            if attempt < max_retries:
                logger.info(f"  Will retry...")
                continue
            else:
                result["error"] = "Timeout (max retries reached)"
                result["status"] = "TIMEOUT"
                return result
                
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            if attempt < max_retries and "connection" in str(e).lower():
                continue
            else:
                result["error"] = str(e)
                return result
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Codex CLI agent on LocBench")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to LocBench JSONL file")
    parser.add_argument("--output_dir", type=str, default="results/codex_agent",
                        help="Directory to save results")
    parser.add_argument("--repos_dir", type=str, default="/tmp/locbench_repos",
                        help="Directory to clone repos")
    parser.add_argument("--repos", type=str, nargs='+', default=None,
                        help="Filter to specific repos (e.g., flask pandas)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="Model to use")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per instance in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show reasoning steps)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum retries per instance on failure")
    parser.add_argument("--retry_delay", type=int, default=60,
                        help="Initial delay between retries in seconds (doubles each retry)")
    parser.add_argument("--delay", type=int, default=0,
                        help="Delay in seconds between consecutive instances (to avoid rate limits)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    
    logger.info(f"Codex Agent Evaluation")
    logger.info(f"=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Retry delay: {args.retry_delay}s (with exponential backoff)")
    logger.info(f"Delay between instances: {args.delay}s")
    logger.info(f"Repos filter: {args.repos}")
    logger.info(f"Output: {args.output_dir}")
    
    # Load dataset
    instances = []
    with open(args.dataset_path) as f:
        for line in f:
            inst = json.loads(line)
            if args.repos:
                repo_name = inst['repo'].split('/')[-1]
                if repo_name not in args.repos:
                    continue
            instances.append(inst)
    
    if args.num_samples:
        instances = instances[:args.num_samples]
    
    logger.info(f"Loaded {len(instances)} instances")
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.repos_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "loc_outputs.jsonl")
    
    # Resume logic
    completed_ids = set()
    existing_results = []
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'FINISHED' and r.get('found_files'):
                    completed_ids.add(r['instance_id'])
                    existing_results.append(r)
        logger.info(f"Resuming: {len(completed_ids)} already completed")
    
    results = existing_results.copy()
    
    # Run evaluation
    import time
    for i, instance in enumerate(tqdm(instances, desc="Evaluating")):
        if instance['instance_id'] in completed_ids:
            logger.info(f"Skipping {instance['instance_id']} (already completed)")
            continue
        
        # Add delay between instances (not before first one)
        if i > 0 and args.delay > 0:
            logger.info(f"  Waiting {args.delay}s before next instance...")
            time.sleep(args.delay)
        
        result = run_codex_agent(
            instance, 
            args.repos_dir, 
            args.model,
            args.timeout,
            logger,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        results.append(result)
        
        # Save incrementally
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
    
    # Summary
    finished = sum(1 for r in results if r['status'] == 'FINISHED')
    with_files = sum(1 for r in results if r.get('found_files'))
    timeouts = sum(1 for r in results if r.get('status') == 'TIMEOUT')
    rate_limited = sum(1 for r in results if r.get('status') == 'RATE_LIMITED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    total_retries = sum(r.get('retries', 0) for r in results)
    avg_files = sum(len(r['found_files']) for r in results) / len(results) if results else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total instances: {len(results)}")
    logger.info(f"  Finished: {finished}")
    logger.info(f"  With files found: {with_files}")
    logger.info(f"  Timeouts: {timeouts}")
    logger.info(f"  Rate limited: {rate_limited}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"Total retries: {total_retries}")
    logger.info(f"Avg files per instance: {avg_files:.2f}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()