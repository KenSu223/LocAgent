# Complete Evaluation Workflow for LocAgent

This guide provides a step-by-step workflow to evaluate LocAgent on the evaluation dataset using your GPT-4.1 endpoint.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Activate your conda environment
   conda activate locagent
   
   # Load API keys and environment variables
   source scripts/env/set_env.sh
   
   # Set PYTHONPATH
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Install Dependencies** (if not already done)
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Prepare Indexes (Optional but Recommended)

Pre-building indexes speeds up the localization process. You can skip this step and indexes will be built on-the-fly, but it's slower.

### Option A: Build Graph Indexes in Batch

```bash
python dependency_graph/batch_build_graph.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --num_processes 50 \
    --download_repo
```

**Parameters:**
- `--dataset`: Choose from `'czlll/Loc-Bench_V1'`, `'czlll/SWE-bench_Lite'`, `'princeton-nlp/SWE-bench_Lite'`
- `--split`: Usually `'test'`
- `--num_processes`: Number of parallel processes (adjust based on your system)
- `--download_repo`: Downloads repositories if not already present

### Option B: Build BM25 Index

```bash
bash scripts/gen_bm25_index.sh
```

### Set Index Directories

After building indexes, set the environment variables:

```bash
export GRAPH_INDEX_DIR='/path/to/your/indexes/graph_index_v2.3'
export BM25_INDEX_DIR='/path/to/your/indexes/BM25_index'
```

**Note:** If you skip index building, LocAgent will generate them during localization (slower).

## Step 2: Run Localization

Run LocAgent on your evaluation dataset:

```bash
# Set output directory
RESULT_DIR="./evaluation_results/gpt4.1"
mkdir -p $RESULT_DIR

# Run localization
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder $RESULT_DIR/location \
    --eval_n_limit 0 \
    --num_processes 4 \
    --use_function_calling \
    --simple_desc \
    --timeout 900
```

**Key Parameters:**
- `--dataset`: Evaluation dataset (`'czlll/Loc-Bench_V1'`, `'czlll/SWE-bench_Lite'`, etc.)
- `--split`: Dataset split (`'test'`)
- `--model`: Your model (`'azure/gpt-4.1'`)
- `--localize`: Enable localization mode
- `--output_folder`: Where results will be saved
- `--eval_n_limit`: Number of instances to evaluate (0 = all, or set a number like 10 for testing)
- `--num_processes`: Parallel processes (start with 4, increase if you have resources)
- `--use_function_calling`: Enable function calling (recommended)
- `--simple_desc`: Use simplified function descriptions
- `--timeout`: Timeout per instance in seconds (900 = 15 minutes)

**For Testing (Small Subset):**
```bash
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder ./test_results \
    --eval_n_limit 10 \
    --num_processes 2 \
    --use_function_calling \
    --simple_desc
```

## Step 3: Merge Results (Optional)

If you ran with `--num_samples > 1`, merge multiple samples:

```bash
python auto_search_main.py \
    --merge \
    --output_folder $RESULT_DIR/location \
    --ranking_method 'mrr'  # or 'majority'
```

This creates a `merged_loc_outputs.jsonl` file with ranked results.

## Step 4: Evaluate Results

### Option A: Using Python Script

Create an evaluation script `evaluate.py`:

```python
import sys
sys.path.append('.')

from evaluation.eval_metric import evaluate_results
import pandas as pd

# Define the mapping between levels and output keys
level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

# Path to your localization results
loc_file = './evaluation_results/gpt4.1/location/loc_outputs.jsonl'
# Or if you merged: './evaluation_results/gpt4.1/location/merged_loc_outputs_mrr.jsonl'

# Evaluate results
results = evaluate_results(
    loc_file=loc_file,
    level2key_dict=level2key_dict,
    dataset='czlll/Loc-Bench_V1',  # Match your dataset
    split='test',
    metrics=['acc', 'ndcg', 'precision', 'recall', 'map'],
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
print("\n")

# Save results to CSV
results.to_csv('./evaluation_results/gpt4.1/evaluation_scores.csv')
print("Results saved to: ./evaluation_results/gpt4.1/evaluation_scores.csv")
```

Run it:
```bash
python evaluate.py
```

### Option B: Using Jupyter Notebook

```bash
jupyter notebook evaluation/run_evaluation.ipynb
```

Then modify the notebook:
```python
from evaluation.eval_metric import evaluate_results

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

# Your results file
loc_file = './evaluation_results/gpt4.1/location/loc_outputs.jsonl'

# Evaluate
results = evaluate_results(
    loc_file,
    level2key_dict,
    dataset='czlll/Loc-Bench_V1',
    split='test',
    metrics=['acc', 'ndcg', 'precision', 'recall', 'map']
)

results
```

## Step 5: Understanding the Metrics

The evaluation calculates metrics at three levels:

### File Level
- **Acc@1, Acc@3, Acc@5**: Accuracy (exact match) at top-k
- **NDCG@1, NDCG@3, NDCG@5**: Normalized Discounted Cumulative Gain
- **Recall@1, Recall@3, Recall@5**: Recall at top-k
- **P@1, P@3, P@5**: Precision at top-k
- **MAP@1, MAP@3, MAP@5**: Mean Average Precision

### Module Level
- Same metrics at k=5, 10

### Function/Entity Level
- Same metrics at k=5, 10

**Key Metrics:**
- **Acc@k**: Percentage of instances where ALL correct locations are in top-k
- **Recall@k**: Percentage of correct locations found in top-k
- **NDCG@k**: Ranking quality metric (higher is better)
- **MAP@k**: Average precision across all instances

## Complete Example Workflow

Here's a complete script that runs everything:

```bash
#!/bin/bash
# complete_evaluation.sh

# Step 1: Setup
source scripts/env/set_env.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 2: Set paths
RESULT_DIR="./evaluation_results/gpt4.1_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

# Step 3: Run localization (test with 10 instances first)
echo "Running localization on 10 test instances..."
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder $RESULT_DIR/location \
    --eval_n_limit 10 \
    --num_processes 2 \
    --use_function_calling \
    --simple_desc \
    --timeout 900

# Step 4: Merge results (if you used --num_samples > 1)
echo "Merging results..."
python auto_search_main.py \
    --merge \
    --output_folder $RESULT_DIR/location \
    --ranking_method 'mrr'

# Step 5: Evaluate
echo "Evaluating results..."
python -c "
import sys
sys.path.append('.')
from evaluation.eval_metric import evaluate_results
import pandas as pd

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

loc_file = '$RESULT_DIR/location/loc_outputs.jsonl'
results = evaluate_results(
    loc_file,
    level2key_dict,
    dataset='czlll/Loc-Bench_V1',
    split='test',
    metrics=['acc', 'ndcg', 'precision', 'recall', 'map']
)

print('\n' + '='*80)
print('EVALUATION RESULTS')
print('='*80)
print(results.to_string())
results.to_csv('$RESULT_DIR/evaluation_scores.csv')
print(f'\nResults saved to: $RESULT_DIR/evaluation_scores.csv')
"

echo "Evaluation complete! Results in: $RESULT_DIR"
```

Make it executable and run:
```bash
chmod +x complete_evaluation.sh
./complete_evaluation.sh
```

## Full Evaluation (All Instances)

Once you've tested with a small subset, run on the full dataset:

```bash
RESULT_DIR="./evaluation_results/gpt4.1_full"
mkdir -p $RESULT_DIR

python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder $RESULT_DIR/location \
    --eval_n_limit 0 \
    --num_processes 8 \
    --use_function_calling \
    --simple_desc \
    --timeout 900
```

**Note:** Full evaluation can take many hours depending on dataset size and number of processes.

## Troubleshooting

### Issue: "No module named 'evaluation'"
**Solution:** Make sure you're in the LocAgent directory and PYTHONPATH is set:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Issue: Results file not found
**Solution:** Check the output folder path. Results are saved as `loc_outputs.jsonl` in your `--output_folder`.

### Issue: Timeout errors
**Solution:** Increase `--timeout` value or reduce `--num_processes` to avoid resource contention.

### Issue: API rate limits
**Solution:** Reduce `--num_processes` to lower concurrent API calls.

## Output Files

After running localization, you'll have:
- `loc_outputs.jsonl`: Raw localization results
- `merged_loc_outputs_mrr.jsonl`: Merged results (if you ran merge)
- `loc_trajs.jsonl`: Full conversation trajectories
- `localize.log`: Execution log
- `args.json`: Arguments used

After evaluation:
- `evaluation_scores.csv`: Detailed metrics in CSV format

## Quick Reference

**Most Important Commands:**

```bash
# 1. Setup
source scripts/env/set_env.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. Run (test with 10 instances)
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder ./results \
    --eval_n_limit 10 \
    --num_processes 2 \
    --use_function_calling \
    --simple_desc

# 3. Evaluate
python -c "
from evaluation.eval_metric import evaluate_results
results = evaluate_results(
    './results/loc_outputs.jsonl',
    {'file': 'found_files', 'module': 'found_modules', 'function': 'found_entities'},
    dataset='czlll/Loc-Bench_V1',
    split='test'
)
print(results)
"
```
