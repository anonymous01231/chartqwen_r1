# ChartQwen-R1

This repository contains an anonymous implementation of ChartQwen-R1, an executable programmatic chart reasoning method for chart question answering.

## Contents

- `code/rewards.py`: GRPO reward functions for executable chart reasoning.
- `scripts/train_grpo.sh`: GRPO training entry script.
- `scripts/eval_chart_datasets.py`: inference script for ChartQA, ChartBench, and ChartX-style datasets.
- `scripts/score_results.py`: scorer for JSON/JSONL inference outputs.
- `scripts/summarize_training_data.py`: utility for summarizing the released training data.
- `scripts/prepare_training_data.py`: utility for normalizing training data paths and checking table fields.
- `scripts/example_pipeline.sh`: example train-evaluate-score pipeline.
- `configs/`: training and evaluation configuration templates.
- `data/chartqwen_r1_train_2876.json`: released training data used for the proposed method.
- `data/prompt.txt`: system prompt used for training and evaluation.

## Environment

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

The training and inference scripts are based on `ms-swift` and PyTorch. Model weights and benchmark images are not included in this anonymous repository.

## Training

Set the model path and run GRPO training:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
TRAIN_DATA=data/chartqwen_r1_train_2876.json \
PROMPT_FILE=data/prompt.txt \
OUTPUT_DIR=outputs/grpo_chartqwen_r1 \
REWARD_PLUGIN=code/rewards.py \
bash scripts/train_grpo.sh
```

The default script uses LoRA fine-tuning, GRPO, vLLM generation, and the four reward functions registered in `code/rewards.py`.

## Evaluation

Run inference on a supported chart QA dataset:

```bash
python scripts/eval_chart_datasets.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --lora outputs/grpo_chartqwen_r1/checkpoint-last \
  --dataset-name ChartQA_h \
  --src-file /path/to/ChartQA/test_human.json \
  --image-dir /path/to/ChartQA/test/png \
  --output outputs/eval_chartqa_h.jsonl \
  --system-prompt data/prompt.txt
```

Then score the generated outputs:

```bash
python scripts/score_results.py \
  --input outputs/eval_chartqa_h.jsonl \
  --output outputs/eval_chartqa_h_scored.json
```

## Data summary

Summarize the released training data:

```bash
python scripts/summarize_training_data.py \
  --data data/chartqwen_r1_train_2876.json
```

Expected summary:

```json
{
  "samples": 2876,
  "images": 2876,
  "messages": 2876,
  "label_types": {
    "text": 848,
    "numeric": 2028
  }
}
```

## Notes

- This anonymous release contains the proposed method implementation, training script, reward functions, evaluation script, scorer, configuration templates, prompt, and released training data.
- It does not include external baseline implementations, model weights, benchmark image files, checkpoints, or private experiment logs.
- Paths in configuration files are placeholders and should be replaced with local paths in the user's environment.
