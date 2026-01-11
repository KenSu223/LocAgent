# Model Setup Guide

Your Azure OpenAI models have been configured and are ready to use!

## Available Models

1. **azure/gpt-4.1** (Default) - Endpoint: `cielara-bkash02.openai.azure.com`
2. **azure/gpt-5.2** - Endpoint: `cielara-research-resource.cognitiveservices.azure.com`
3. **azure/gpt-4.1-mini** - Endpoint: `cielara-research-resource.cognitiveservices.azure.com`
4. **azure/deepseek-v3.2** - Endpoint: `bowen-mjiwqwei-northcentralus.services.ai.azure.com`

## Quick Start

1. **Load environment variables:**
   ```bash
   source scripts/env/set_env.sh
   ```

2. **Run LocAgent with default model (GPT-4.1):**
   ```bash
   python auto_search_main.py \
       --dataset 'czlll/SWE-bench_Lite' \
       --split 'test' \
       --model 'azure/gpt-4.1' \
       --localize \
       --output_folder ./results
   ```

3. **Use a different model:**
   ```bash
   # Use GPT-5.2
   python auto_search_main.py --model 'azure/gpt-5.2' --localize --output_folder ./results
   
   # Use GPT-4.1-mini (faster, cheaper)
   python auto_search_main.py --model 'azure/gpt-4.1-mini' --localize --output_folder ./results
   
   # Use DeepSeek-V3.2
   python auto_search_main.py --model 'azure/deepseek-v3.2' --localize --output_folder ./results
   ```

## Configuration Details

API keys should live in `scripts/env/set_env.local.sh` (this file is git-ignored). The configuration automatically:
- Maps model names to the correct Azure endpoints
- Uses the correct deployment names for each model
- Handles authentication for all your Azure resources

## Model Selection

The default model is **azure/gpt-4.1**. You can change it by:
- Using the `--model` argument when running the script
- Editing the default in `auto_search_main.py` (line 597)

## Notes

- Store API keys in `scripts/env/set_env.local.sh` and keep `scripts/env/set_env.sh` committed without secrets
- The code automatically routes to the correct Azure endpoint based on the model name
- Deployment names are automatically handled (e.g., "DeepSeek-V3.2" for deepseek-v3.2)
