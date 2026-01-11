# Quick Start: Evaluate LocAgent with GPT-4.1

## 🚀 Fastest Way (All-in-One Script)

```bash
# 1. Setup environment
source scripts/env/set_env.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. Run complete evaluation (test with 10 instances)
./run_evaluation.sh czlll/Loc-Bench_V1 test 10 2

# Parameters:
#   $1: dataset (default: czlll/Loc-Bench_V1)
#   $2: split (default: test)
#   $3: number of instances (default: 10, use 0 for all)
#   $4: number of processes (default: 2)
```

## 📋 Step-by-Step Manual Workflow

### Step 1: Run Localization

```bash
# Test with 10 instances first
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
```

### Step 2: Evaluate Results

```bash
python evaluate.py \
    --loc_file ./results/loc_outputs.jsonl \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test'
```

## 📊 Understanding Results

The evaluation outputs metrics at three levels:

- **File Level**: Acc@1, Acc@3, Acc@5, NDCG@1-5, Recall@1-5, etc.
- **Module Level**: Metrics at k=5, 10
- **Function Level**: Metrics at k=5, 10

**Key Metrics:**
- **Acc@k**: Accuracy - all correct locations in top-k
- **Recall@k**: Percentage of correct locations found
- **NDCG@k**: Ranking quality (0-1, higher is better)

## 🔧 Full Evaluation (All Instances)

```bash
# Run on full dataset (this will take hours)
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder ./full_results \
    --eval_n_limit 0 \
    --num_processes 8 \
    --use_function_calling \
    --simple_desc

# Then evaluate
python evaluate.py \
    --loc_file ./full_results/loc_outputs.jsonl \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --output_csv ./full_results/scores.csv
```

## 📁 Output Files

After running, you'll find:
- `loc_outputs.jsonl` - Raw localization results
- `merged_loc_outputs_mrr.jsonl` - Merged/ranked results
- `evaluation_scores.csv` - Evaluation metrics
- `localize.log` - Execution log

## ⚡ Quick Test (5 minutes)

```bash
# Test with just 1 instance
python auto_search_main.py \
    --dataset 'czlll/Loc-Bench_V1' \
    --split 'test' \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder ./test \
    --eval_n_limit 1 \
    --num_processes 1 \
    --use_function_calling \
    --simple_desc

python evaluate.py --loc_file ./test/loc_outputs.jsonl
```

## 📚 More Details

See `EVALUATION_WORKFLOW.md` for complete documentation.
