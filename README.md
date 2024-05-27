# HPTO Training Framework

## Env
```
conda create -n hpto python=3.10 -y
conda activate hpto
pip install packaging
pip install torch==2.1.2
pip install -e .
pip install -e ".[train]"

if model_name == phi_3:
    pip install transformers==4.41.1
else:
    pip install git+https://github.com/huggingface/transformers@v4.36.2
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
```


# Exps
HPT
```
stage-1-alignment
bash scripts/v1_5/release/hpt/1_mm_align.sh

stage-2-sft
bash scripts/v1_5/release/hpt/3_sft.sh
```

VILA
```
stage-1-alignment
bash scripts/v1_5/release/3b/1_mm_align.sh

stage-2-pretrain(2 node)
bash scripts/v1_5/release/3b/2_pretrain_0.sh
bash scripts/v1_5/release/3b/2_pretrain_1.sh

stage-3-sft
bash scripts/v1_5/release/3b/3_sft.sh
```