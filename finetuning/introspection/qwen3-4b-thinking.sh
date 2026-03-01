#!/bin/bash

export PATH="/workspace/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
cd /workspace

torchrun --nproc_per_node=1 --master_port=29510 \
    -m openrlhf.cli.train_sft \
    --save_path /workspace/loras/qwen3-introspection/$1 \
    --eval_steps 50 \
    --save_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/distilled/qwen3-4b-thinking-$1 \
    --dataset /workspace/OpenCharacterTraining/data/sft_data/qwen3-4b-thinking/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 8192 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --attn_implementation sdpa \
    --gradient_checkpointing

if [ $? -ne 0 ]; then
    echo "error: training failed"
    exit 1
fi
