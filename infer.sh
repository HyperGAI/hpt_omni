python llava/eval/run_vila.py \
    --model-path /export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b/stage2/checkpoint-14000 \
    --conv-mode llama_3 \
    --query "<image>\n Please extract the text in the image." \
    --image-file './images/ocr_0.jpg'