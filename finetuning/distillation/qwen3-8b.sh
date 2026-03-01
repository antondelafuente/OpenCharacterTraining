#!/bin/bash

export PATH="/workspace/venv/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
cd /workspace

torchrun --nproc_per_node=1 --master_port=29550 \
    -m openrlhf.cli.train_dpo \
    --save_path /workspace/loras/qwen3-distillation/$1 \
    --eval_steps 25 \
    --save_steps 25 \
    --max_ckpt_num 5 \
    --logging_steps 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/qwen3-8b \
    --dataset /workspace/OpenCharacterTraining/data/dpo/qwen3-8b/$1.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --length_normalize \
    --max_len 4096 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --attn_implementation sdpa \
    --gradient_checkpointing

if [ $? -ne 0 ]; then
    echo "error: training failed"
    exit 1
fi
