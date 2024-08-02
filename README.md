# HPTO Training Framework

## Release Note
[0702] hpto 1.0.0
- model_path: /export/share/yucheng/models/hpto_1.0
- hpto wheel: /export/share/yucheng/hpt/hpt_omni/dist/hpt_omni-1.0.0-py3-none-any.whl


## Env
```
conda create -n hpto python=3.10 -y
conda activate hpto
pip install packaging
pip install torch==2.1.2
pip install -e .
pip install -e ".[train]"
pip install transformers==4.41.1
pip uninstall transformer-engine
pip uninstall apex

# for llama3.1, should update transformers version:
pip install transformers==4.43.1
```

## Exps
HPT1.5 Edge(Phi-3-mini-4k-instruct)

```
stage-1-alignment
bash scripts/train/hpt_phi3/1_align.sh

stage-2-sft(2 node)
bash scripts/train/hpt_phi3/2_sft.sh
```

VILA-3B(Sheared-LLaMA-2.7B)
```
stage-1-alignment
bash scripts/train/vila_3b/1_mm_align.sh

stage-2-pretrain(3 node)
bash scripts/train/vila_3b/2_pretrain.sh

stage-3-sft
bash scripts/train/vila_3b/3_sft.sh
```

## Eval
https://github.com/HyperGAI/HPTEvalKit