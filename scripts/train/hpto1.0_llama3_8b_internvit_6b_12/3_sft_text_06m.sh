#!/bin/bash

n_node=8
nproc_per_node=8
MASTER_ADDR=172.29.251.36
CURRENT_RANK=$1

# Train
OUTPUT='hpt_llama3_8b_internvit6b_fix_lr/stage3_with_text_06m'
BS=8
VERSION='llama_3_fix'
MM_PROJECTOR='mlp_downsample'

# Data
DATA_MIXTURE='hpt_v41+vatex+doc_downstream+doc_reason25k+doc_local4k+hpt_text_06m'

# Pretrained model
BASE_MODEL_PATH='/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b_internvit6b_fix_lr/stage2'


torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version $VERSION \
    --data_mixture $DATA_MIXTURE \
    --vision_tower /export/share/models/InternViT-6B-448px-V1-2 \
    --mm_vision_select_feature patch \
    --mm_projector $MM_PROJECTOR \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --image_size 448 \
    --mm_vit_lr 1e-5 \
    --report_to none