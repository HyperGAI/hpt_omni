python -W ignore llava/eval/run_vila_mi.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
    --conv-mode llama_3 \
    --query "<image>\n Please describe the traffic condition." \
    --image-dir "/export/share/yucheng/hpt/data/mi_imgs/1"


# python -W ignore llava/eval/run_vila_mi.py \
#     --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
#     --conv-mode llama_3 \
#     --image-file "/export/share/yucheng/hpt/data/mi_imgs/0/0314e72bf7e964b539c20a5d34643500d5b924baf.png"
