# PASeg

## 1. conda environment install

```
conda create -n PathSeg python=3.12
conda activate PathSeg

conda install -c conda-forge scikit-image opencv pandas pillow numpy

conda install -c conda-forge openslide openslide-python

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers==4.46.1

pip install pycocotools matplotlib scikit-learn

pip install accelerate==0.26.0

conda install -c conda-forge opencv

conda install -c conda-forge albumentations
```
