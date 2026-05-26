#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen3-VL-8B-Instruct"}
LORA_PATH=${LORA_PATH:-"$ROOT_DIR/outputs/grpo_chartqwen_r1/checkpoint-last"}

python "$ROOT_DIR/scripts/summarize_training_data.py" \
  --data "$ROOT_DIR/data/chartqwen_r1_train_2876.json"

MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$ROOT_DIR/data/chartqwen_r1_train_2876.json" \
PROMPT_FILE="$ROOT_DIR/data/prompt.txt" \
OUTPUT_DIR="$ROOT_DIR/outputs/grpo_chartqwen_r1" \
REWARD_PLUGIN="$ROOT_DIR/code/rewards.py" \
bash "$ROOT_DIR/scripts/train_grpo.sh"

python "$ROOT_DIR/scripts/eval_chart_datasets.py" \
  --model "$MODEL_PATH" \
  --lora "$LORA_PATH" \
  --dataset-name ChartQA_h \
  --src-file "/path/to/ChartQA/test_human.json" \
  --image-dir "/path/to/ChartQA/test/png" \
  --output "$ROOT_DIR/outputs/eval_chartqa_h.jsonl" \
  --system-prompt "$ROOT_DIR/data/prompt.txt"

python "$ROOT_DIR/scripts/score_results.py" \
  --input "$ROOT_DIR/outputs/eval_chartqa_h.jsonl" \
  --output "$ROOT_DIR/outputs/eval_chartqa_h_scored.json"
