#!/usr/bin/env python3
"""
Evaluate OpenAI Codex CLI agent on MULocBench/LocBench.
Parallel processing with round-robin across multiple Azure endpoints.

python codex/run_codex_parallel.py \
    --dataset_path /home/tsu25/LocAgent/evaluation/mulocbench_2/mulocbench.jsonl \
    --output_dir results/codex_agent_parallel \
    --repos_dir /home/tsu25/LocAgent/mulocbench_repos \
    --repos transformers \
    --model gpt-5.2 \
    --timeout 300 \
    --max_retries 3 \
    --retry_delay 30 \
    --endpoints_file /home/tsu25/LocAgent/codex/azure_endpoints.json \
    --workers 2 \
    --num_samples 2 \
    -v
"""


import json
import subprocess
import os
import re
import logging
import threading
import queue
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


class EndpointPool:
    """Thread-safe round-robin endpoint pool."""
    
    def __init__(self, endpoints: List[Dict[str, str]]):
        self.endpoints = endpoints
        self.index = 0
        self.lock = threading.Lock()
    
    def get_endpoint(self) -> Dict[str, str]:
        """Get next endpoint in round-robin fashion (thread-safe)."""
        with self.lock:
            endpoint = self.endpoints[self.index]
            self.index = (self.index + 1) % len(self.endpoints)
            return endpoint


class ResultWriter:
    """Thread-safe result writer."""
    
    def __init__(self, output_path: str, initial_results: List[dict] = None):
        self.output_path = output_path
        self.results = initial_results or []
        self.lock = threading.Lock()
    
    def add_result(self, result: dict):
        """Add a result and save to file (thread-safe)."""
        with self.lock:
            self.results.append(result)
            with open(self.output_path, 'w') as f:
                for r in self.results:
                    f.write(json.dumps(r) + '\n')
    
    def get_results(self) -> List[dict]:
        with self.lock:
            return self.results.copy()


def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """Setup logging to both file and console (thread-safe)."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("codex_eval")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - always verbose
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def clone_repo_at_commit(repo: str, commit: str, target_dir: str, logger: logging.Logger) -> bool:
    """Clone a repo and checkout specific commit."""
    if os.path.exists(target_dir):
        # Already exists, just checkout the commit
        logger.info(f"  Checking out {commit[:8]}...")
        try:
            subprocess.run(
                ["git", "fetch", "--all"],
                cwd=target_dir,
                capture_output=True,
                timeout=120
            )
            subprocess.run(
                ["git", "checkout", commit, "-f"],
                cwd=target_dir,
                capture_output=True,
                check=True,
                timeout=60
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=target_dir,
                capture_output=True,
                timeout=60
            )
            return True
        except Exception as e:
            logger.error(f"  Failed to checkout: {e}")
            return False
    else:
        # Clone fresh
        logger.info(f"  Cloning {repo}...")
        try:
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", target_dir],
                capture_output=True,
                check=True,
                timeout=300
            )
            subprocess.run(
                ["git", "checkout", commit, "-f"],
                cwd=target_dir,
                capture_output=True,
                check=True,
                timeout=60
            )
            return True
        except Exception as e:
            logger.error(f"  Failed to clone: {e}")
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


def is_rate_limit_error(response: str) -> bool:
    """Check if response indicates a rate limit error."""
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded",
        "tokens per min",
        "TPM",
        "RPM"
    ]
    response_lower = response.lower()
    return any(indicator.lower() in response_lower for indicator in rate_limit_indicators)


def is_retriable_error(response: str) -> bool:
    """Check if error is retriable."""
    retriable_indicators = [
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection reset",
        "stream disconnected",
        "response.failed"
    ]
    response_lower = response.lower()
    return any(indicator.lower() in response_lower for indicator in retriable_indicators)


def parse_codex_response(response: str, logger: logging.Logger) -> Tuple[List[str], List[str], List[str]]:
    """Parse Codex CLI JSONL response to extract found files and entities."""
    found_files = []
    found_modules = []
    found_entities = []
    
    logger.info(f"  Parsing response...")
    
    # Parse JSONL events
    for line in response.strip().split('\n'):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event.get('type', '')
            
            if event_type == 'item.completed':
                item = event.get('item', {})
                item_type = item.get('type', '')
                
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
                    for match in re.finditer(r'^\.?/?([a-zA-Z0-9_/\-]+\.py):\d+:', output, re.MULTILINE):
                        fpath = match.group(1)
                        # Filter out test fixtures and example paths
                        if '/tmp/' in fpath or 'foobar' in fpath or '__pycache__' in fpath:
                            continue
                        if fpath not in found_files:
                            found_files.append(fpath)
                            logger.debug(f"    [FOUND FILE from grep] {fpath}")
                
                elif item_type in ('assistant_message', 'agent_message'):
                    text = item.get('text', '')
                    logger.debug(f"    [ASSISTANT] {text[:500]}...")
                    
                    # Parse FILES section
                    files_match = re.search(r'FILES:\s*\n((?:[-*]\s*[^\n]+\n?)+)', text, re.IGNORECASE)
                    if files_match:
                        files_section = files_match.group(1)
                        for line in files_section.strip().split('\n'):
                            line = line.strip()
                            if line.startswith(('-', '*')):
                                fpath = line.lstrip('-* `').rstrip('`').strip()
                                if fpath and fpath not in found_files:
                                    found_files.append(fpath)
                                    logger.debug(f"    [FOUND FILE from response] {fpath}")
                    
                    # Parse FUNCTIONS section
                    funcs_match = re.search(r'FUNCTIONS:\s*\n((?:[-*]\s*[^\n]+\n?)+)', text, re.IGNORECASE)
                    if funcs_match:
                        funcs_section = funcs_match.group(1)
                        for line in funcs_section.strip().split('\n'):
                            line = line.strip()
                            if line.startswith(('-', '*')):
                                entity = line.lstrip('-* `').rstrip('`').strip()
                                if entity and ':' in entity and entity not in found_entities:
                                    found_entities.append(entity)
                                    logger.debug(f"    [FOUND ENTITY from response] {entity}")
            
            elif event_type == 'error':
                error_msg = event.get('message', '')
                logger.warning(f"    [ERROR] {error_msg}")
            
            elif event_type == 'turn.failed':
                error = event.get('error', {})
                error_msg = error.get('message', str(error))
                logger.warning(f"    [TURN FAILED] {error_msg}")
                
        except json.JSONDecodeError:
            continue
    
    # Fallback: regex search for Python files in entire response
    if not found_files:
        logger.debug("    [FALLBACK] Using regex to find files...")
        for match in re.finditer(r'([a-zA-Z0-9_/\-]+\.py)', response):
            fpath = match.group(1)
            # Filter out common false positives
            if fpath in found_files:
                continue
            if fpath.startswith('python'):
                continue
            if fpath.startswith('/tmp/'):
                continue
            if fpath.startswith('/path/to/'):
                continue
            if fpath.startswith('/usr/'):
                continue
            if fpath.startswith('foobar/'):
                continue
            if '__pycache__' in fpath:
                continue
            # Must have a directory component to be a real project file
            if '/' not in fpath:
                continue
            # Skip if it looks like a generic example path
            if 'example' in fpath.lower() or 'hello.py' in fpath:
                continue
                
            found_files.append(fpath)
            logger.debug(f"    [FOUND FILE from fallback] {fpath}")
    
    # Modules are same as files for now
    found_modules = found_files.copy()
    
    return found_files, found_modules, found_entities


def run_codex_agent(
    instance: dict, 
    repos_dir: str, 
    model: str,
    timeout: int,
    logger: logging.Logger,
    endpoint: Dict[str, str],
    max_retries: int = 3,
    retry_delay: int = 60
) -> dict:
    """Run Codex CLI agent on a single instance with retry logic."""
    import time
    
    instance_id = instance['instance_id']
    repo = instance['repo']
    commit = instance['base_commit']
    problem = instance['problem_statement']
    
    endpoint_name = endpoint['name']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Instance: {instance_id}")
    logger.info(f"Repo: {repo} @ {commit[:8]}")
    logger.info(f"Endpoint: {endpoint_name}")
    logger.info(f"{'='*60}")
    
    result = {
        "instance_id": instance_id,
        "found_files": [],
        "found_modules": [],
        "found_entities": [],
        "status": "FAILED",
        "error": None,
        "raw_response": "",
        "retries": 0,
        "endpoint": endpoint_name
    }
    
    # Setup repo - use endpoint-specific directory to avoid conflicts
    repo_dir = os.path.join(repos_dir, f"{repo.replace('/', '_')}_{endpoint_name}")
    if not clone_repo_at_commit(repo, commit, repo_dir, logger):
        result["error"] = "Failed to clone repo"
        return result
    
    # Build prompt
    prompt = build_prompt(problem)
    logger.debug(f"  Problem: {problem[:200]}...")
    
    # Retry loop
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"  [{endpoint_name}] Retry {attempt}/{max_retries} after {retry_delay}s delay...")
            time.sleep(retry_delay)
            # Exponential backoff: double the delay for next retry
            retry_delay = min(retry_delay * 2, 300)  # Cap at 5 minutes
        
        # Setup environment
        env = os.environ.copy()
        env["AZURE_OPENAI_API_KEY"] = endpoint["api_key"]
        
        # Write temporary config.toml
        temp_config_path = os.path.join(repo_dir, ".codex_temp_config.toml")
        config_content = f'''model = "{model}"
model_provider = "azure"
model_reasoning_effort = "medium"
personality = "pragmatic"
model_reasoning_summary = "concise"
model_verbosity = "low"

[model_providers.azure]
name = "Azure OpenAI"
base_url = "{endpoint['base_url']}"
env_key = "AZURE_OPENAI_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 600000
request_max_retries = 10
stream_max_retries = 30
'''
        with open(temp_config_path, 'w') as f:
            f.write(config_content)
        
        cmd = [
            "codex", "exec",
            "--json",
            "--sandbox", "danger-full-access",
            "--model", model,
            "--config", temp_config_path,
            prompt
        ]
        
        logger.info(f"  [{endpoint_name}] Running Codex agent (attempt {attempt + 1}/{max_retries + 1})...")
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            # Cleanup temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            response = proc.stdout
            stderr = proc.stderr or ""
            combined_output = response + stderr
            
            result["raw_response"] = response
            result["retries"] = attempt
            
            # Log stderr if any
            if stderr:
                logger.debug(f"  [{endpoint_name}] [STDERR] {stderr[:500]}")
            
            # Check for rate limit errors
            if is_rate_limit_error(combined_output):
                logger.warning(f"  [{endpoint_name}] Rate limit hit!")
                if attempt < max_retries:
                    logger.info(f"  [{endpoint_name}] Will retry after delay...")
                    continue
                else:
                    result["error"] = "Rate limit exceeded (max retries reached)"
                    result["status"] = "RATE_LIMITED"
                    return result
            
            # Check for other retriable errors (like stream disconnection)
            if is_retriable_error(combined_output) and not response.strip():
                logger.warning(f"  [{endpoint_name}] Retriable error detected")
                if attempt < max_retries:
                    continue
                else:
                    result["error"] = "Retriable error (max retries reached)"
                    return result
            
            # Parse the response
            found_files, found_modules, found_entities = parse_codex_response(response, logger)
            
            result["found_files"] = found_files
            result["found_modules"] = found_modules
            result["found_entities"] = found_entities
            result["status"] = "FINISHED"
            
            logger.info(f"  [{endpoint_name}] Results: {len(found_files)} files, {len(found_modules)} modules, {len(found_entities)} entities")
            logger.info(f"  [{endpoint_name}] Files: {found_files[:5]}{'...' if len(found_files) > 5 else ''}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.warning(f"  [{endpoint_name}] Timeout after {timeout}s")
            if attempt < max_retries:
                continue
            result["error"] = f"Timeout after {timeout}s"
            result["status"] = "TIMEOUT"
            # Cleanup temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return result
            
        except Exception as e:
            logger.error(f"  [{endpoint_name}] Error: {e}")
            if attempt < max_retries:
                continue
            result["error"] = str(e)
            # Cleanup temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return result
    
    return result


def worker(
    instance: dict,
    repos_dir: str,
    model: str,
    timeout: int,
    logger: logging.Logger,
    endpoint_pool: EndpointPool,
    result_writer: ResultWriter,
    max_retries: int,
    retry_delay: int,
    pbar: tqdm
) -> dict:
    """Worker function for parallel processing."""
    # Get an endpoint from the pool
    endpoint = endpoint_pool.get_endpoint()
    
    # Run the agent
    result = run_codex_agent(
        instance=instance,
        repos_dir=repos_dir,
        model=model,
        timeout=timeout,
        logger=logger,
        endpoint=endpoint,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Save result
    result_writer.add_result(result)
    
    # Update progress bar
    pbar.update(1)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Codex CLI agent on LocBench (Parallel)")
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
    parser.add_argument("--endpoints_file", type=str, required=True,
                        help="JSON file with Azure endpoints for parallel processing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of endpoints)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    
    # Load endpoints
    with open(args.endpoints_file) as f:
        endpoints = json.load(f)
    
    num_endpoints = len(endpoints)
    num_workers = args.workers or num_endpoints
    
    logger.info(f"Codex Agent Evaluation (Parallel)")
    logger.info(f"=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Retry delay: {args.retry_delay}s (with exponential backoff)")
    logger.info(f"Repos filter: {args.repos}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Endpoints: {num_endpoints}")
    for ep in endpoints:
        logger.info(f"  - {ep['name']}: {ep['base_url']}")
    logger.info(f"Parallel workers: {num_workers}")
    
    # Initialize endpoint pool
    endpoint_pool = EndpointPool(endpoints)
    
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
    
    # Filter instances to process
    instances_to_process = [inst for inst in instances if inst['instance_id'] not in completed_ids]
    logger.info(f"Instances to process: {len(instances_to_process)}")
    
    # Initialize result writer
    result_writer = ResultWriter(output_path, existing_results)
    
    # Run evaluation in parallel
    with tqdm(total=len(instances_to_process), desc="Evaluating") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for instance in instances_to_process:
                future = executor.submit(
                    worker,
                    instance=instance,
                    repos_dir=args.repos_dir,
                    model=args.model,
                    timeout=args.timeout,
                    logger=logger,
                    endpoint_pool=endpoint_pool,
                    result_writer=result_writer,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    pbar=pbar
                )
                futures.append(future)
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker error: {e}")
    
    # Summary
    results = result_writer.get_results()
    finished = sum(1 for r in results if r['status'] == 'FINISHED')
    with_files = sum(1 for r in results if r.get('found_files'))
    timeouts = sum(1 for r in results if r.get('status') == 'TIMEOUT')
    rate_limited = sum(1 for r in results if r.get('status') == 'RATE_LIMITED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    total_retries = sum(r.get('retries', 0) for r in results)
    avg_files = sum(len(r['found_files']) for r in results) / len(results) if results else 0
    
    # Count by endpoint
    endpoint_counts = {}
    for r in results:
        ep = r.get('endpoint', 'unknown')
        endpoint_counts[ep] = endpoint_counts.get(ep, 0) + 1
    
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
    logger.info(f"By endpoint:")
    for ep, count in endpoint_counts.items():
        logger.info(f"  {ep}: {count}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()