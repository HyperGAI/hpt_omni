# HPT Training Framework

## Env
```
conda create -n hpto python=3.10 -y
conda activate hpto
pip install packaging
pip install torch==2.1.2
pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.36.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
```