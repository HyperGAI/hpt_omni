# HPTO Training Framework

## Env
```
conda create -n hpto python=3.10 -y
conda activate hpto
pip install packaging
pip install torch==2.1.2
pip install -e .
pip install -e ".[train]"
pip install transformers==4.41.1
```

## Exps
HPT1.5 Edge(Phi3-Mini)

```
stage-1-alignment
bash scripts/train/hpt_phi3/1_align.sh

stage-2-sft
bash scripts/train/hpt_phi3/2_sft_0.sh
```

VILA
```
stage-1-alignment
bash scripts/train/vila_3b/1_mm_align.sh

stage-2-pretrain(2 node)
bash scripts/train/vila_3b/2_pretrain_0.sh

stage-3-sft
bash scripts/train/vila_3b/3_sft.sh
```

## Eval
https://github.com/HyperGAI/HPTEvalKit