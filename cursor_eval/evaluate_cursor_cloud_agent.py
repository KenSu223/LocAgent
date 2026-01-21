"""
Async Cursor Cloud Agent Evaluation for MULocBench

Two-phase evaluation that separates launching from result collection:
1. Launch phase: Quickly submit all agents (~0.5s each)
2. Collect phase: Poll and gather results (run later or in background)

This allows you to launch 100+ agents in under a minute, then collect results
whenever the agents finish (typically 1-10 minutes each, running in parallel).

Usage:
    # Phase 1: Launch all agents (fast - just HTTP POSTs)
    python evaluate_cursor_cloud_agent.py launch \
        --dataset_path data/mulocbench/mulocbench.jsonl \
        --output_dir results/cursor \
        --api_key YOUR_CURSOR_API_KEY \
        --num_samples 100

    # Phase 2: Collect results (run anytime later)
    python evaluate_cursor_cloud_agent.py collect \
        --output_dir results/cursor \
        --api_key YOUR_CURSOR_API_KEY

    # Or collect without waiting for pending agents
    python evaluate_cursor_cloud_agent.py collect \
        --output_dir results/cursor \
        --api_key YOUR_CURSOR_API_KEY \
        --no_wait

    # Check status of launched agents
    python evaluate_cursor_cloud_agent.py status \
        --output_dir results/cursor \
        --api_key YOUR_CURSOR_API_KEY

API Documentation: https://cursor.com/docs/cloud-agent/api/endpoints
"""

import argparse
import asyncio
import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CURSOR_API_BASE = "https://api.cursor.com/v0"


class CursorCloudClient:
    """Client for Cursor Cloud Agents API"""
    
    def __init__(self, api_key: str):
        # Cursor uses Basic Auth with API key + trailing colon, base64 encoded
        auth_string = f"{api_key}:"
        encoded = base64.b64encode(auth_string.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def verify_api_key(self, session: aiohttp.ClientSession) -> Dict:
        """Verify API key and get account info"""
        url = f"{CURSOR_API_BASE}/me"
        async with session.get(url, headers=self.headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"API key verification failed: {resp.status} - {text}")
            return await resp.json()
    
    async def list_models(self, session: aiohttp.ClientSession) -> List[str]:
        """Get available models"""
        url = f"{CURSOR_API_BASE}/models"
        async with session.get(url, headers=self.headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("models", [])
            return []
    
    async def launch_agent(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        repo_url: str,
        ref: str = "main",
        model: Optional[str] = None
    ) -> str:
        """
        Launch a Cloud Agent (non-blocking).
        
        Returns:
            Agent ID (use this to check status and get results later)
        """
        url = f"{CURSOR_API_BASE}/agents"
        
        payload = {
            "prompt": {"text": prompt},
            "source": {
                "repository": repo_url,
                "ref": ref
            }
        }
        
        if model:
            payload["model"] = model
        
        async with session.post(url, headers=self.headers, json=payload) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise Exception(f"Failed to launch agent: {resp.status} - {text}")
            
            data = await resp.json()
            return data.get("id")
    
    async def get_agent_status(
        self,
        session: aiohttp.ClientSession,
        agent_id: str
    ) -> Dict:
        """Get agent status and details"""
        url = f"{CURSOR_API_BASE}/agents/{agent_id}"
        async with session.get(url, headers=self.headers) as resp:
            if resp.status != 200:
                return {"status": "ERROR", "error": f"HTTP {resp.status}"}
            return await resp.json()
    
    async def get_conversation(
        self,
        session: aiohttp.ClientSession,
        agent_id: str
    ) -> List[Dict]:
        """Get agent conversation history"""
        url = f"{CURSOR_API_BASE}/agents/{agent_id}/conversation"
        async with session.get(url, headers=self.headers) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("messages", [])
    
    async def stop_agent(
        self,
        session: aiohttp.ClientSession,
        agent_id: str
    ) -> None:
        """Stop a running agent"""
        url = f"{CURSOR_API_BASE}/agents/{agent_id}/stop"
        async with session.post(url, headers=self.headers) as resp:
            pass  # Best effort
    
    async def delete_agent(
        self,
        session: aiohttp.ClientSession,
        agent_id: str
    ) -> None:
        """Delete an agent"""
        url = f"{CURSOR_API_BASE}/agents/{agent_id}"
        async with session.delete(url, headers=self.headers) as resp:
            pass  # Best effort


def create_localization_prompt(problem_statement: str) -> str:
    """
    Create a prompt for the Cursor agent to perform code localization.
    """
    return f"""You are performing code localization for a GitHub issue. Your task is to identify 
the specific code locations (files, classes, and functions) that need to be modified to resolve this issue.

## Issue Description
{problem_statement}

## Instructions
1. Explore the repository structure to understand the codebase
2. Identify ALL files that need to be modified to resolve this issue
3. For each file, identify the specific functions or methods that need changes
4. Do NOT make any actual code changes - only identify locations

## Required Output Format
After exploring the codebase, provide your findings in this exact format:

LOCALIZATION RESULTS:
FILES:
- path/to/file1.py
- path/to/file2.py

FUNCTIONS:
- path/to/file1.py:function_name
- path/to/file1.py:ClassName.method_name
- path/to/file2.py:another_function

REASONING:
Brief explanation of why these locations need changes.

Remember: Only identify locations, do not modify any code."""


def parse_agent_response(conversation: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Parse the agent's conversation to extract localized files and functions.
    
    Returns:
        Tuple of (files, functions)
    """
    files = []
    functions = []
    
    # Combine all assistant messages
    full_response = ""
    for msg in conversation:
        if msg.get("type") == "assistant_message":
            full_response += msg.get("text", "") + "\n"
    
    # Try to parse structured output - FILES section
    files_match = re.search(
        r'FILES:\s*\n((?:[-•*]\s*[\w/\.]+\s*\n?)+)',
        full_response,
        re.IGNORECASE
    )
    if files_match:
        files_text = files_match.group(1)
        files = re.findall(r'[-•*]\s*([\w/\.]+\.py)', files_text)
    
    # Try to parse structured output - FUNCTIONS section
    funcs_match = re.search(
        r'FUNCTIONS:\s*\n((?:[-•*]\s*[\w/\.:]+\s*\n?)+)',
        full_response,
        re.IGNORECASE
    )
    if funcs_match:
        funcs_text = funcs_match.group(1)
        functions = re.findall(r'[-•*]\s*([\w/\.]+:\w+(?:\.\w+)?)', funcs_text)
    
    # Fallback: extract any file:function patterns from full response
    if not functions:
        functions = re.findall(r'([\w/]+\.py:\w+(?:\.\w+)?)', full_response)
    
    # Fallback: extract any .py files mentioned
    if not files:
        files = list(set(re.findall(r'([\w/]+\.py)', full_response)))
    
    return list(set(files)), list(set(functions))


# =============================================================================
# PHASE 1: LAUNCH
# =============================================================================

async def launch_all_agents(
    dataset_path: str,
    output_dir: str,
    api_key: str,
    num_samples: Optional[int] = None,
    repos_filter: Optional[List[str]] = None,
    rate_limit: float = 0.5,
    model: Optional[str] = None,
    use_main_branch: bool = False,
    branch: str = "main"
):
    """
    Phase 1: Launch all agents as quickly as possible.
    
    This just sends POST requests to start agents - doesn't wait for completion.
    Saves a manifest file with agent IDs for later collection.
    
    Args:
        dataset_path: Path to MULocBench JSONL file
        output_dir: Directory to save manifest and results
        api_key: Cursor API key
        num_samples: Limit number of samples (None = all)
        repos_filter: List of repo names to include (e.g., ["scikit-learn", "flask"])
        rate_limit: Seconds between launches (to avoid rate limits)
        model: Optional model name (None = auto select)
        use_main_branch: If True, use branch name instead of commit SHAs
        branch: Branch name to use (default: main)
    """
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    instances = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    # Filter by repository if specified
    if repos_filter:
        # Normalize filter terms (lowercase, handle variations)
        filter_terms = [r.lower().replace("-", "").replace("_", "") for r in repos_filter]
        
        def matches_filter(repo: str) -> bool:
            # repo format: "owner/repo_name"
            repo_name = repo.split("/")[-1].lower().replace("-", "").replace("_", "")
            return any(term in repo_name or repo_name in term for term in filter_terms)
        
        original_count = len(instances)
        instances = [inst for inst in instances if matches_filter(inst.get("repo", ""))]
        logger.info(f"Filtered to {len(instances)}/{original_count} instances matching repos: {repos_filter}")
        
        # Show which repos matched
        matched_repos = set(inst.get("repo", "") for inst in instances)
        logger.info(f"Matched repositories: {sorted(matched_repos)}")
    
    if num_samples:
        instances = instances[:num_samples]
    
    logger.info(f"Will launch {len(instances)} agents")
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    client = CursorCloudClient(api_key)
    
    # Track launched agents
    launched = []
    
    async with aiohttp.ClientSession() as session:
        # Verify API key first
        logger.info("Verifying API key...")
        try:
            account_info = await client.verify_api_key(session)
            logger.info(f"Authenticated as: {account_info.get('userEmail', 'unknown')}")
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return
        
        # Launch agents
        start_time = time.time()
        
        for i, inst in enumerate(instances):
            instance_id = inst["instance_id"]
            repo = inst.get("repo", "")
            base_commit = inst.get("base_commit", "main")
            
            if "/" not in repo:
                logger.warning(f"Skipping {instance_id}: invalid repo format '{repo}'")
                launched.append({
                    "instance_id": instance_id,
                    "agent_id": None,
                    "error": f"Invalid repo format: {repo}"
                })
                continue
            
            repo_url = f"https://github.com/{repo}.git"
            prompt = create_localization_prompt(inst["problem_statement"])
            
            # Determine which ref to use
            if use_main_branch:
                ref_to_use = branch
            else:
                ref_to_use = base_commit
                if not base_commit or base_commit == "main":
                    logger.warning(f"{instance_id}: No specific commit SHA, using default branch")
            
            try:
                agent_id = await client.launch_agent(
                    session,
                    prompt=prompt,
                    repo_url=repo_url,
                    ref=ref_to_use,
                    model=model
                )
                
                launched.append({
                    "instance_id": instance_id,
                    "agent_id": agent_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "launched_at": time.time(),
                    "edit_functions": inst.get("edit_functions", [])  # Ground truth
                })
                
                logger.info(f"[{i+1}/{len(instances)}] Launched agent {agent_id} for {instance_id}")
                
            except Exception as e:
                logger.error(f"[{i+1}/{len(instances)}] Failed to launch for {instance_id}: {e}")
                launched.append({
                    "instance_id": instance_id,
                    "agent_id": None,
                    "error": str(e)
                })
            
            # Rate limiting
            if i < len(instances) - 1:
                await asyncio.sleep(rate_limit)
        
        elapsed = time.time() - start_time
    
    # Save manifest
    manifest = {
        "launched_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "dataset_path": dataset_path,
        "total": len(launched),
        "successful": sum(1 for x in launched if x.get("agent_id")),
        "failed": sum(1 for x in launched if not x.get("agent_id")),
        "agents": launched
    }
    
    manifest_path = os.path.join(output_dir, "launch_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("LAUNCH COMPLETE")
    print("="*60)
    print(f"Total instances:    {manifest['total']}")
    print(f"Successfully launched: {manifest['successful']}")
    print(f"Failed to launch:   {manifest['failed']}")
    print(f"Time elapsed:       {elapsed:.1f}s")
    print(f"Manifest saved to:  {manifest_path}")
    print(f"\nNext step: Run 'collect' command to gather results")
    print("="*60)


# =============================================================================
# PHASE 2: COLLECT
# =============================================================================

async def collect_results(
    output_dir: str,
    api_key: str,
    poll_pending: bool = True,
    poll_interval: int = 30,
    max_wait: int = 3600
):
    """
    Phase 2: Collect results from previously launched agents.
    
    Can be run immediately after launch or hours later.
    
    Args:
        output_dir: Directory containing launch_manifest.json
        api_key: Cursor API key
        poll_pending: If True, wait for pending agents; if False, just collect finished ones
        poll_interval: Seconds between status polls
        max_wait: Maximum seconds to wait for all agents
    """
    # Load manifest
    manifest_path = os.path.join(output_dir, "launch_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run 'launch' command first")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    agents = [a for a in manifest["agents"] if a.get("agent_id")]
    logger.info(f"Collecting results for {len(agents)} agents...")
    
    client = CursorCloudClient(api_key)
    results = []
    
    async with aiohttp.ClientSession() as session:
        pending = list(agents)
        start_time = time.time()
        iteration = 0
        
        while pending:
            iteration += 1
            elapsed = time.time() - start_time
            
            if elapsed >= max_wait:
                logger.warning(f"Max wait time ({max_wait}s) reached")
                break
            
            still_pending = []
            finished_this_round = 0
            
            for agent in pending:
                try:
                    status_resp = await client.get_agent_status(session, agent["agent_id"])
                    agent_status = status_resp.get("status", "UNKNOWN")
                except Exception as e:
                    logger.warning(f"Error getting status for {agent['agent_id']}: {e}")
                    agent_status = "ERROR"
                
                if agent_status in ("FINISHED", "FAILED", "STOPPED", "ERROR"):
                    # Collect conversation and parse results
                    conversation = await client.get_conversation(session, agent["agent_id"])
                    files, functions = parse_agent_response(conversation)
                    
                    results.append({
                        "instance_id": agent["instance_id"],
                        "agent_id": agent["agent_id"],
                        "repo": agent.get("repo", ""),
                        "base_commit": agent.get("base_commit", ""),
                        "status": agent_status,
                        "found_files": files,
                        "found_edit_locs": functions,
                        "edit_functions": agent.get("edit_functions", []),
                        "raw_conversation": conversation
                    })
                    
                    finished_this_round += 1
                    logger.info(f"Collected {agent['instance_id']}: {agent_status} "
                               f"(files={len(files)}, funcs={len(functions)})")
                else:
                    still_pending.append(agent)
            
            pending = still_pending
            
            if finished_this_round > 0:
                logger.info(f"Round {iteration}: {finished_this_round} finished, "
                           f"{len(pending)} still pending")
            
            if pending:
                if poll_pending:
                    logger.info(f"Waiting {poll_interval}s before next poll...")
                    await asyncio.sleep(poll_interval)
                else:
                    # Don't wait - just record pending status and exit
                    logger.info(f"Recording {len(pending)} agents as PENDING (--no_wait mode)")
                    for agent in pending:
                        results.append({
                            "instance_id": agent["instance_id"],
                            "agent_id": agent["agent_id"],
                            "repo": agent.get("repo", ""),
                            "status": "PENDING",
                            "found_files": [],
                            "found_edit_locs": [],
                            "edit_functions": agent.get("edit_functions", [])
                        })
                    break
    
    # Save results
    output_path = os.path.join(output_dir, "loc_outputs.jsonl")
    with open(output_path, 'w') as f:
        for r in results:
            # Don't save raw conversation to JSONL (too verbose)
            r_clean = {k: v for k, v in r.items() if k != "raw_conversation"}
            f.write(json.dumps(r_clean) + "\n")
    
    logger.info(f"Results saved to {output_path}")
    
    # Compute and display metrics
    compute_metrics(results, output_dir)


# =============================================================================
# STATUS CHECK
# =============================================================================

async def check_status(output_dir: str, api_key: str):
    """
    Check status of launched agents without collecting full results.
    """
    manifest_path = os.path.join(output_dir, "launch_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    agents = [a for a in manifest["agents"] if a.get("agent_id")]
    
    client = CursorCloudClient(api_key)
    status_counts = {}
    
    async with aiohttp.ClientSession() as session:
        for agent in agents:
            try:
                status_resp = await client.get_agent_status(session, agent["agent_id"])
                status = status_resp.get("status", "UNKNOWN")
            except:
                status = "ERROR"
            
            status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\n" + "="*60)
    print("AGENT STATUS")
    print("="*60)
    print(f"Total agents: {len(agents)}")
    for status, count in sorted(status_counts.items()):
        pct = count / len(agents) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    print("="*60)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(results: List[Dict], output_dir: str):
    """Compute and display evaluation metrics"""
    
    finished = [r for r in results if r.get("status") == "FINISHED"]
    
    metrics = {
        "total": len(results),
        "finished": len(finished),
        "failed": sum(1 for r in results if r.get("status") in ("FAILED", "ERROR")),
        "pending": sum(1 for r in results if r.get("status") in ("PENDING", "RUNNING", "CREATING")),
        "file_level": {"acc@1": 0, "acc@3": 0, "acc@5": 0},
        "function_level": {"acc@5": 0, "acc@10": 0},
    }
    
    if finished:
        for r in finished:
            gt_files = set(f.split(":")[0] for f in r.get("edit_functions", []))
            gt_functions = set(r.get("edit_functions", []))
            
            pred_files = set(r.get("found_files", []))
            pred_functions = set(r.get("found_edit_locs", []))
            
            # File-level accuracy
            if gt_files and gt_files.issubset(pred_files):
                metrics["file_level"]["acc@1"] += 1
                metrics["file_level"]["acc@3"] += 1
                metrics["file_level"]["acc@5"] += 1
            
            # Function-level accuracy
            if gt_functions and gt_functions.issubset(pred_functions):
                metrics["function_level"]["acc@5"] += 1
                metrics["function_level"]["acc@10"] += 1
        
        # Normalize
        n = len(finished)
        for level in ["file_level", "function_level"]:
            for k in metrics[level]:
                metrics[level][k] = metrics[level][k] / n * 100
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Total instances:  {metrics['total']}")
    print(f"Finished:         {metrics['finished']}")
    print(f"Failed/Error:     {metrics['failed']}")
    print(f"Pending:          {metrics['pending']}")
    
    if finished:
        print(f"\nFile-level Accuracy (n={len(finished)}):")
        print(f"  Acc@1: {metrics['file_level']['acc@1']:.1f}%")
        print(f"  Acc@3: {metrics['file_level']['acc@3']:.1f}%")
        print(f"  Acc@5: {metrics['file_level']['acc@5']:.1f}%")
        print(f"\nFunction-level Accuracy:")
        print(f"  Acc@5:  {metrics['function_level']['acc@5']:.1f}%")
        print(f"  Acc@10: {metrics['function_level']['acc@10']:.1f}%")
    else:
        print("\nNo finished agents to compute metrics")
    
    print("="*60)
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cursor Cloud Agents on MULocBench (Async Version)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch all agents (fast - just sends POST requests)"
    )
    launch_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to MULocBench JSONL file"
    )
    launch_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cursor_evaluation",
        help="Directory to save manifest and results"
    )
    launch_parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Cursor API key"
    )
    launch_parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    launch_parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific repos (e.g., --repos scikit-learn flask requests)"
    )
    launch_parser.add_argument(
        "--rate_limit",
        type=float,
        default=0.5,
        help="Seconds between agent launches (default: 0.5)"
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-select)"
    )
    launch_parser.add_argument(
        "--use_main_branch",
        action="store_true",
        help="Use default branch instead of commit SHAs (required since Cursor doesn't support commit refs)"
    )
    launch_parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch name to use when --use_main_branch is set (default: main). Try 'master' for older repos."
    )
    
    # Collect command
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect results from launched agents"
    )
    collect_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cursor_evaluation",
        help="Directory containing launch_manifest.json"
    )
    collect_parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Cursor API key"
    )
    collect_parser.add_argument(
        "--no_wait",
        action="store_true",
        help="Don't wait for pending agents, just collect finished ones"
    )
    collect_parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between status polls (default: 30)"
    )
    collect_parser.add_argument(
        "--max_wait",
        type=int,
        default=3600,
        help="Maximum seconds to wait for agents (default: 3600)"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check status of launched agents"
    )
    status_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cursor_evaluation",
        help="Directory containing launch_manifest.json"
    )
    status_parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Cursor API key"
    )
    
    args = parser.parse_args()
    
    if args.command == "launch":
        asyncio.run(launch_all_agents(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            api_key=args.api_key,
            num_samples=args.num_samples,
            repos_filter=args.repos,
            rate_limit=args.rate_limit,
            model=args.model,
            use_main_branch=args.use_main_branch,
            branch=args.branch
        ))
    
    elif args.command == "collect":
        asyncio.run(collect_results(
            output_dir=args.output_dir,
            api_key=args.api_key,
            poll_pending=not args.no_wait,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait
        ))
    
    elif args.command == "status":
        asyncio.run(check_status(
            output_dir=args.output_dir,
            api_key=args.api_key
        ))


if __name__ == "__main__":
    main()