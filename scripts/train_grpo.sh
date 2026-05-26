#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen3-VL-8B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"../data/chartqwen_r1_train_2876.json"}
PROMPT_FILE=${PROMPT_FILE:-"../data/prompt.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"../outputs/grpo_chartqwen_r1"}
REWARD_PLUGIN=${REWARD_PLUGIN:-"../code/rewards.py"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  --external_plugins "$REWARD_PLUGIN" \
  --reward_funcs PythonExecutable_reward AnswerAccuracy_reward TableCorrectness_reward VariableStructure_reward \
  --reward_weights 1 3 2 1 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --vllm_max_model_len 4096 \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --torch_dtype bfloat16 \
  --dataset "$TRAIN_DATA" \
  --system "$PROMPT_FILE" \
  --load_from_cache_file true \
  --max_completion_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --learning_rate 5e-6 \
  --gradient_accumulation_steps 2 \
  --save_strategy steps \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 7 \
  --logging_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.01 \
  --dataloader_num_workers 4 \
  --num_generations 4 \
  --temperature 1.0 \
  --deepspeed zero2 \
  --log_completions true \
  --num_iterations 1 \
  --async_generate false \
  --beta 0.005
