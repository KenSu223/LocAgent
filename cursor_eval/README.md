# Cursor MULocBench Pipeline (Local CLI Agent + LocAgent Metrics)

This folder evaluates Cursor’s local CLI agent on MULocBench and scores results
with LocAgent-style metrics.

## What each script does

- `convert_mulocbench_to_locbench.py`
  - Downloads MULocBench and converts it into a LocBench-style JSONL with
    `instance_id`, `repo`, `base_commit`, `problem_statement`, and `edit_functions`.
- `evaluate_cursor_cli_local.py` (**run_cursor_cli_local** step)
  - Clones each repo at `base_commit`, runs the Cursor CLI agent locally, and
    writes predictions to `loc_outputs.jsonl`.
- `run_locagent_eval.py`
  - Computes LocAgent-compatible metrics from Cursor’s `loc_outputs.jsonl`
    against the converted MULocBench file.

## End-to-end: convert → run Cursor CLI → evaluate

### 1) Convert MULocBench to LocBench-style JSONL

```bash
python cursor_eval/convert_mulocbench_to_locbench.py \
  --output_dir evaluation/datasets/mulocbench \
  --output_format jsonl
```

Output:
- `evaluation/datasets/mulocbench/mulocbench.jsonl`

### 2) Run Cursor CLI local agent (run_cursor_cli_local)

**export the cursor API key first before running this**

```bash
python cursor_eval/evaluate_cursor_cli_local.py \
  --dataset_path evaluation/datasets/mulocbench/mulocbench.jsonl \
  --output_dir evaluation_results/cursor_cli \
  --repos scikit-learn flask localstack transformers pandas \
  --num_samples 0 \
  --num_workers 1 \
  --model claude-4-sonnet
```

What it does:
- Clones each repo at `base_commit` into a local cache.
- Runs `cursor-agent` non-interactively to localize files/functions.
- Writes predictions to:
  - `evaluation_results/cursor_cli/loc_outputs.jsonl`

Notes:
- `--num_samples 0` = run all instances (set a number to limit).
- Use low `--num_workers` to avoid rate limits.

### 3) Evaluate with LocAgent metrics

```bash
python cursor_eval/run_locagent_eval.py \
  --loc_file evaluation_results/cursor_cli/loc_outputs.jsonl \
  --gt_file evaluation/datasets/mulocbench/mulocbench.jsonl \
  --only_predicted \
  --output_dir evaluation_results/cursor_cli
```

What it does:
- Loads ground truth from `edit_functions` in `mulocbench.jsonl`.
- Computes file/module/function metrics using LocAgent’s definitions.
- Writes a metrics JSON/CSV to the output folder.

## Metrics definition (LocAgent-style)

- **Acc@k**: strict accuracy — **all** GT locations must appear in top‑k.
- **Recall@k**: fraction of GT locations found in top‑k.
- **NDCG@k**: ranking quality (higher is better).
- **Precision@k**: fraction of top‑k predictions that are correct.

Levels:
- **File**: match `file.py`
- **Module**: match `file.py:Class` or `file.py:function`
- **Function**: match `file.py:Class.method` or `file.py:function`

## Output format (Cursor CLI)

`loc_outputs.jsonl` entries include:
- `instance_id`
- `repo`, `base_commit`
- `found_files`, `found_modules`, `found_entities`
- `raw_response`, timing metadata

## Minimal pipeline summary

1) Convert: `convert_mulocbench_to_locbench.py`  
2) Run Cursor CLI: `evaluate_cursor_cli_local.py`  
3) Evaluate: `run_locagent_eval.py`

## Troubleshooting

### Git clone fails
- Ensure `git` is installed and you have internet access
- Some repos may require authentication - set `GIT_ASKPASS` or use SSH

### API rate limits
- Reduce `--num_workers` to avoid rate limits
- Add delays between requests (modify the script)

### Memory issues
- Use `--num_samples` to limit evaluation size
- Clear `--repo_cache_dir` periodically

## Citation

If you use this evaluation pipeline, please cite:

```bibtex
@article{mulocbench2024,
    title={MULocBench: A Benchmark for Localizing Code and Non-Code Issues in Software Projects},
    author={...},
    year={2024}
}

@inproceedings{locagent2025,
    title={LocAgent: Graph-Guided LLM Agents for Code Localization},
    author={Chen, Zhaoling and Tang, Robert and Deng, Gangda and Wu, Fang and Wu, Jialong and Jiang, Zhiwei and Prasanna, Viktor and Cohan, Arman and Wang, Xingyao},
    booktitle={ACL},
    year={2025}
}
```
