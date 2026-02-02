#!/bin/bash
set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$(pwd)

REPOS="${REPOS:-scikit-learn,flask,localstack,transformers,pandas}"
DATASET="${DATASET:-evaluation/datasets/mulocbench/mulocbench.jsonl}"
GT_FILE="${GT_FILE:-evaluation/datasets/mulocbench/gt_location.jsonl}"
OUT_DIR="${OUT_DIR:-evaluation_results/mulocbench_subset}"
MODEL="${MODEL:-azure/gpt-4.1}"
TIMEOUT="${TIMEOUT:-900}"
NUM_PROCESSES=1
NUM_SAMPLES=1

if [[ ! -f "$DATASET" || ! -f "$GT_FILE" ]]; then
  echo "Missing MULocBench files. Run: python scripts/convert_mulocbench.py --output_dir evaluation/datasets/mulocbench"
  exit 1
fi

mkdir -p "$OUT_DIR"

python scripts/filter_mulocbench_subset.py \
  --input_dataset "$DATASET" \
  --input_gt "$GT_FILE" \
  --repos "$REPOS" \
  --output_dir "$OUT_DIR"

SUBSET_DATASET="$OUT_DIR/mulocbench_subset.jsonl"
SUBSET_GT="$OUT_DIR/gt_location.jsonl"
INSTANCE_IDS="$OUT_DIR/selected_instance_ids.json"
DATASET_NAME="$(basename "$SUBSET_DATASET" .jsonl)"
GRAPH_VERSION="$(python - <<'PY'
from dependency_graph.build_graph import VERSION
print(VERSION)
PY
)"

# python dependency_graph/batch_build_graph.py \
#   --dataset "$SUBSET_DATASET" \
#   --split train \
#   --num_processes "$NUM_PROCESSES" \
#   --download_repo \
#   --repo_path playground/build_graph \
#   --index_dir index_data \
#   --instance_id_path "$INSTANCE_IDS"

# python build_bm25_index.py \
#   --dataset "$SUBSET_DATASET" \
#   --split train \
#   --num_processes "$NUM_PROCESSES" \
#   --download_repo \
#   --repo_path playground/build_graph \
#   --index_dir index_data \
#   --instance_id_path "$INSTANCE_IDS"

if [ -z "${GRAPH_INDEX_DIR:-}" ]; then
  export GRAPH_INDEX_DIR="index_data/${DATASET_NAME}/graph_index_${GRAPH_VERSION}"
fi
if [ -z "${BM25_INDEX_DIR:-}" ]; then
  export BM25_INDEX_DIR="index_data/${DATASET_NAME}/BM25_index"
fi

python auto_search_main.py \
  --dataset "$SUBSET_DATASET" \
  --split train \
  --model "$MODEL" \
  --localize \
  --output_folder "$OUT_DIR/location" \
  --eval_n_limit 0 \
  --num_processes "$NUM_PROCESSES" \
  --num_samples "$NUM_SAMPLES" \
  --use_function_calling \
  --simple_desc \
  --timeout "$TIMEOUT"

python scripts/evaluate_mulocbench.py \
  --loc_file "$OUT_DIR/location/loc_outputs.jsonl" \
  --gt_file "$SUBSET_GT" \
  --output_csv "$OUT_DIR/evaluation_scores.csv"
