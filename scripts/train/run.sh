#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1

n_node=1
nproc_per_node=1
MASTER_ADDR=172.29.251.36
CURRENT_RANK=$1

# Train
OUTPUT='hpt_llama3_8b_internvit6b_fix_lr/stage2'
BS=8
VERSION='llama_3_fix'
MM_PROJECTOR='mlp_downsample'

# Data
DATA_MIXTURE='obelics'
# DATA_MIXTURE='multi_grained_text_localization_1m'

# Pretrained model
BASE_MODEL_PATH='/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b_internvit6b/stage1'


torchrun --nnodes=$n_node --nproc_per_node=$nproc_per_node --master_port=25001 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version $VERSION \
    --data_mixture $DATA_MIXTURE \
    --vision_tower /export/share/models/InternViT-6B-448px-V1-2 \
    --mm_vision_select_feature patch \
    --mm_projector $MM_PROJECTOR \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --image_size 448 \
    --mm_vit_lr 5e-6 \
    --report_to none
