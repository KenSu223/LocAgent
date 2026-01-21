"""
MULocBench Evaluation Script

Evaluates LLM performance on code localization tasks using MULocBench dataset.
Outputs results compatible with LocAgent's evaluation pipeline.

Usage:
    python evaluate_mulocbench.py \
        --dataset_path evaluation/datasets/mulocbench/mulocbench.jsonl \
        --output_dir results/cursor_evaluation \
        --api_key YOUR_API_KEY \
        --api_base https://api.openai.com/v1 \
        --model gpt-4o \
        --num_samples 50
"""

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import aiohttp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Result of a single localization attempt"""
    instance_id: str
    found_files: List[str] = field(default_factory=list)
    found_classes: List[str] = field(default_factory=list)  # file.py:ClassName
    found_functions: List[str] = field(default_factory=list)  # file.py:func or file.py:Class.func
    raw_response: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass 
class EvaluationConfig:
    """Configuration for evaluation run"""
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    num_samples: Optional[int] = None
    num_workers: int = 5
    repo_cache_dir: str = "./repo_cache"
    timeout: int = 120
    include_file_content: bool = True
    max_files_in_context: int = 50


class RepoManager:
    """Manages cloning and checking out repositories"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_repo_path(self, repo: str, base_commit: str) -> Path:
        """Get path to repo at specific commit"""
        safe_name = repo.replace("/", "__")
        return self.cache_dir / f"{safe_name}__{base_commit[:8]}"
    
    def clone_and_checkout(self, repo: str, base_commit: str) -> Path:
        """Clone repo and checkout specific commit"""
        repo_path = self.get_repo_path(repo, base_commit)
        
        if repo_path.exists():
            logger.debug(f"Using cached repo: {repo_path}")
            return repo_path
        
        logger.info(f"Cloning {repo} at {base_commit[:8]}...")
        
        # Clone with minimal depth first, then fetch specific commit
        github_url = f"https://github.com/{repo}.git"
        
        try:
            # Clone repo
            subprocess.run(
                ["git", "clone", "--depth", "1", github_url, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=300
            )
            
            # Fetch the specific commit
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=120
            )
            
            # Checkout the commit
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=60
            )
            
            return repo_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout cloning {repo}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning {repo}: {e.stderr.decode() if e.stderr else str(e)}")
            raise
    
    def get_file_tree(self, repo_path: Path, max_depth: int = 3) -> str:
        """Get file tree structure of repository"""
        result = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", "*/\\.*"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        files = sorted(result.stdout.strip().split("\n"))
        # Remove ./ prefix
        files = [f[2:] if f.startswith("./") else f for f in files if f]
        
        return "\n".join(files[:200])  # Limit to 200 files
    
    def get_file_content(self, repo_path: Path, file_path: str) -> Optional[str]:
        """Get content of a specific file"""
        full_path = repo_path / file_path
        try:
            if full_path.exists() and full_path.stat().st_size < 100000:  # Max 100KB
                return full_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
        return None


class LLMClient:
    """Async client for LLM API calls"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def localize(
        self,
        session: aiohttp.ClientSession,
        problem_statement: str,
        file_tree: str,
        file_contents: Optional[Dict[str, str]] = None
    ) -> Tuple[str, int]:
        """
        Call LLM to localize code that needs to be changed.
        
        Returns:
            Tuple of (response_text, tokens_used)
        """
        
        system_prompt = """You are an expert software engineer tasked with code localization.
Given a GitHub issue description and a repository structure, identify the specific code locations 
that need to be modified to resolve the issue.

You must output your response in the following JSON format:
{
    "files": ["path/to/file1.py", "path/to/file2.py"],
    "functions": ["path/to/file1.py:function_name", "path/to/file2.py:ClassName.method_name"],
    "classes": ["path/to/file1.py:ClassName"],
    "reasoning": "Brief explanation of why these locations need changes"
}

Rules:
1. List ALL files that need to be modified
2. For functions, use format: "filepath:function_name" for standalone functions
3. For methods, use format: "filepath:ClassName.method_name"
4. For classes (if the whole class needs changes), use format: "filepath:ClassName"
5. Be specific - identify exact functions/methods, not just files
6. Only include locations that actually need code changes
7. Do NOT include test files unless the issue specifically mentions tests need fixing"""

        user_content = f"""## Issue Description
{problem_statement}

## Repository File Structure
```
{file_tree}
```
"""
        
        if file_contents:
            user_content += "\n## Relevant File Contents\n"
            for fpath, content in file_contents.items():
                # Truncate long files
                if len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"
                user_content += f"\n### {fpath}\n```python\n{content}\n```\n"
        
        user_content += "\n\nBased on the issue description and repository structure, identify the code locations that need to be modified. Output your response as JSON."

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        url = f"{self.config.api_base}/chat/completions"
        
        async with session.post(url, headers=self.headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"API error {resp.status}: {error_text}")
            
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return content, tokens


def parse_llm_response(response: str) -> Dict[str, List[str]]:
    """Parse LLM response to extract locations"""
    
    # Try to extract JSON from response
    import re
    
    # Find JSON block
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "files": data.get("files", []),
                "functions": data.get("functions", []),
                "classes": data.get("classes", [])
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: extract file paths and function names using patterns
    files = re.findall(r'[\w/]+\.py', response)
    functions = re.findall(r'[\w/]+\.py:\w+(?:\.\w+)?', response)
    
    return {
        "files": list(set(files)),
        "functions": list(set(functions)),
        "classes": []
    }


async def evaluate_instance(
    instance: Dict,
    config: EvaluationConfig,
    repo_manager: RepoManager,
    session: aiohttp.ClientSession,
    llm_client: LLMClient
) -> LocalizationResult:
    """Evaluate a single instance"""
    
    instance_id = instance["instance_id"]
    result = LocalizationResult(instance_id=instance_id)
    
    try:
        # Clone/checkout repo
        repo_path = repo_manager.clone_and_checkout(
            instance["repo"],
            instance["base_commit"]
        )
        
        # Get file tree
        file_tree = repo_manager.get_file_tree(repo_path)
        
        # Optionally get some file contents
        file_contents = {}
        if config.include_file_content:
            # Get content of files that might be relevant based on keywords in problem statement
            problem_lower = instance["problem_statement"].lower()
            for line in file_tree.split("\n")[:config.max_files_in_context]:
                file_name = Path(line).name.lower()
                # Simple heuristic: include files whose names appear in the problem
                if any(word in problem_lower for word in file_name.replace(".py", "").replace("_", " ").split()):
                    content = repo_manager.get_file_content(repo_path, line)
                    if content:
                        file_contents[line] = content
        
        # Call LLM
        start_time = time.time()
        response, tokens = await llm_client.localize(
            session,
            instance["problem_statement"],
            file_tree,
            file_contents if file_contents else None
        )
        result.latency_ms = (time.time() - start_time) * 1000
        result.tokens_used = tokens
        result.raw_response = response
        
        # Parse response
        parsed = parse_llm_response(response)
        result.found_files = parsed["files"]
        result.found_functions = parsed["functions"]
        result.found_classes = parsed["classes"]
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"Error evaluating {instance_id}: {e}")
    
    return result


def convert_to_locagent_format(result: LocalizationResult, instance: Dict) -> Dict:
    """
    Convert result to LocAgent-compatible format for evaluation.
    
    LocAgent expects:
    - instance_id
    - found_files (list of file paths)
    - found_edit_locs or found_functions (list of file:function entries)
    """
    return {
        "instance_id": result.instance_id,
        "repo": instance.get("repo", ""),
        "base_commit": instance.get("base_commit", ""),
        
        # Predictions
        "found_files": result.found_files,
        "found_related_locs": result.found_classes,  # Classes/modules
        "found_edit_locs": result.found_functions,   # Functions to edit
        
        # Ground truth (from dataset)
        "edit_functions": instance.get("edit_functions", []),
        
        # Metadata
        "raw_response": result.raw_response,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "tokens_used": result.tokens_used
    }


async def run_evaluation(
    dataset_path: str,
    output_dir: str,
    config: EvaluationConfig
):
    """Run full evaluation"""
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    instances = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    if config.num_samples:
        instances = instances[:config.num_samples]
    
    logger.info(f"Evaluating {len(instances)} instances")
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    repo_manager = RepoManager(config.repo_cache_dir)
    llm_client = LLMClient(config)
    
    results = []
    
    # Run evaluation with concurrency control
    semaphore = asyncio.Semaphore(config.num_workers)
    
    async def bounded_evaluate(instance):
        async with semaphore:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as session:
                return await evaluate_instance(
                    instance, config, repo_manager, session, llm_client
                )
    
    # Process with progress bar
    tasks = [bounded_evaluate(inst) for inst in instances]
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await coro
        # Find corresponding instance
        instance = next(i for i in instances if i["instance_id"] == result.instance_id)
        results.append(convert_to_locagent_format(result, instance))
    
    # Save results
    output_path = os.path.join(output_dir, "loc_outputs.jsonl")
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    logger.info(f"Saved results to {output_path}")
    
    # Compute quick metrics
    compute_metrics(results, output_dir)
    
    return results


def compute_metrics(results: List[Dict], output_dir: str):
    """Compute and save evaluation metrics"""
    
    metrics = {
        "total": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "file_level": {"acc@1": 0, "acc@3": 0, "acc@5": 0},
        "function_level": {"acc@5": 0, "acc@10": 0},
    }
    
    for r in results:
        if r.get("error"):
            continue
        
        gt_files = set(f.split(":")[0] for f in r.get("edit_functions", []))
        gt_functions = set(r.get("edit_functions", []))
        
        pred_files = set(r.get("found_files", []))
        pred_functions = set(r.get("found_edit_locs", []))
        
        # File-level accuracy (all GT files found in top-k predictions)
        if gt_files and gt_files.issubset(pred_files):
            metrics["file_level"]["acc@1"] += 1
            metrics["file_level"]["acc@3"] += 1
            metrics["file_level"]["acc@5"] += 1
        
        # Function-level accuracy
        if gt_functions and gt_functions.issubset(pred_functions):
            metrics["function_level"]["acc@5"] += 1
            metrics["function_level"]["acc@10"] += 1
    
    # Normalize
    valid = metrics["total"] - metrics["errors"]
    if valid > 0:
        for level in ["file_level", "function_level"]:
            for k in metrics[level]:
                metrics[level][k] = metrics[level][k] / valid * 100
    
    # Add averages
    metrics["avg_latency_ms"] = sum(r.get("latency_ms", 0) for r in results) / len(results)
    metrics["avg_tokens"] = sum(r.get("tokens_used", 0) for r in results) / len(results)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Total instances: {metrics['total']}")
    print(f"Errors: {metrics['errors']}")
    print(f"\nFile-level Accuracy:")
    print(f"  Acc@1: {metrics['file_level']['acc@1']:.1f}%")
    print(f"  Acc@3: {metrics['file_level']['acc@3']:.1f}%")
    print(f"  Acc@5: {metrics['file_level']['acc@5']:.1f}%")
    print(f"\nFunction-level Accuracy:")
    print(f"  Acc@5: {metrics['function_level']['acc@5']:.1f}%")
    print(f"  Acc@10: {metrics['function_level']['acc@10']:.1f}%")
    print(f"\nAvg Latency: {metrics['avg_latency_ms']:.0f}ms")
    print(f"Avg Tokens: {metrics['avg_tokens']:.0f}")
    print("="*50)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on MULocBench code localization"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to MULocBench JSONL file (in LocBench format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory to save results"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for LLM service"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API (OpenAI-compatible)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use"
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
        default=5,
        help="Number of concurrent workers"
    )
    parser.add_argument(
        "--repo_cache_dir",
        type=str,
        default="./repo_cache",
        help="Directory to cache cloned repositories"
    )
    parser.add_argument(
        "--no_file_content",
        action="store_true",
        help="Don't include file contents in prompt"
    )
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        repo_cache_dir=args.repo_cache_dir,
        include_file_content=not args.no_file_content
    )
    
    asyncio.run(run_evaluation(args.dataset_path, args.output_dir, config))


if __name__ == "__main__":
    main()
