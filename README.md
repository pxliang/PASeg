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

## 2. Inference

### a. do inference without bbx

```
dataset_name=TCGA-COAD

python3 inference.py \
  --infer_vis_dir ./results/${dataset_name} \
  --checkpoint_file checkpoints/pytorch_model.bin \
  --image_file /data/TCGA-COAD/20x_images/TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.tif \
  --bbx_random 1 \
  --class_names "Tumor" "Stroma"

```

### b. do inference with bbx

```
dataset_name=BRCA

python3 inference_bbx.py \
  --infer_vis_dir ./results/${dataset_name} \
  --json_file ./idx_to_names/${dataset_name}.json \
  --checkpoint_file ./checkpoints/pytorch_model.bin \
  --image_dir ./examples/${dataset_name}/images \
  --mask_dir ./examples/${dataset_name}/masks \
  --bbx_random 0 
```
