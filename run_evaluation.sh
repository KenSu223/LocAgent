#!/bin/bash
# Complete evaluation workflow script for LocAgent
# This script runs localization and evaluation in one go

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LocAgent Evaluation Workflow${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Setup
echo -e "${YELLOW}Step 1: Setting up environment...${NC}"
if [ -f "scripts/env/set_env.sh" ]; then
    source scripts/env/set_env.sh
    echo -e "${GREEN}✓ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠ Warning: scripts/env/set_env.sh not found. Make sure API keys are set.${NC}"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo -e "${GREEN}✓ PYTHONPATH set${NC}\n"

# Step 2: Parse arguments
DATASET=${1:-'czlll/Loc-Bench_V1'}
SPLIT=${2:-'test'}
NUM_INSTANCES=${3:-10}
NUM_PROCESSES=${4:-2}
OUTPUT_DIR=${5:-"./evaluation_results/gpt4.1_$(date +%Y%m%d_%H%M%S)"}

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Dataset: ${DATASET}"
echo -e "  Split: ${SPLIT}"
echo -e "  Instances: ${NUM_INSTANCES} (0 = all)"
echo -e "  Processes: ${NUM_PROCESSES}"
echo -e "  Output: ${OUTPUT_DIR}\n"

# Create output directory
mkdir -p "${OUTPUT_DIR}/location"

# Step 3: Run localization
echo -e "${YELLOW}Step 2: Running localization...${NC}"
echo -e "This may take a while depending on the number of instances...\n"

python auto_search_main.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --model 'azure/gpt-4.1' \
    --localize \
    --output_folder "${OUTPUT_DIR}/location" \
    --eval_n_limit "${NUM_INSTANCES}" \
    --num_processes "${NUM_PROCESSES}" \
    --use_function_calling \
    --simple_desc \
    --timeout 900

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Localization failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Localization complete${NC}\n"

# Step 4: Merge results (if multiple samples)
echo -e "${YELLOW}Step 3: Merging results...${NC}"
python auto_search_main.py \
    --merge \
    --output_folder "${OUTPUT_DIR}/location" \
    --ranking_method 'mrr'

echo -e "${GREEN}✓ Results merged${NC}\n"

# Step 5: Evaluate
echo -e "${YELLOW}Step 4: Evaluating results...${NC}"

LOC_FILE="${OUTPUT_DIR}/location/loc_outputs.jsonl"
if [ -f "${OUTPUT_DIR}/location/merged_loc_outputs_mrr.jsonl" ]; then
    LOC_FILE="${OUTPUT_DIR}/location/merged_loc_outputs_mrr.jsonl"
    echo -e "Using merged results: ${LOC_FILE}"
fi

python evaluate.py \
    --loc_file "${LOC_FILE}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output_csv "${OUTPUT_DIR}/evaluation_scores.csv"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Evaluation failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Evaluation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved in: ${OUTPUT_DIR}"
echo -e "  - Localization results: ${OUTPUT_DIR}/location/"
echo -e "  - Evaluation scores: ${OUTPUT_DIR}/evaluation_scores.csv"
echo -e "\n"
