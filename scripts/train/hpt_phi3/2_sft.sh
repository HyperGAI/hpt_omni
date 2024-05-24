#!/bin/bash

n_node=1
MASTER_ADDR=172.29.201.25
CURRENT_RANK=0
BASE_MODEL_PATH='/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt15_phi3_mlp_stage1'
OUTPUT='hpt15_phi3_mlp_stage2'
bs=16

torchrun --nnodes=$n_node --nproc_per_node=1 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version llama_3 \
    --data_mixture hpt_v41_text01 \
    --vision_tower /export/share/models/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp \
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
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --image_size 490 \
    --vflan_no_system_prompt True \
    --report_to none