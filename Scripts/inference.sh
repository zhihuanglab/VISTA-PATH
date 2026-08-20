#!/bin/bash
# Prompt-free whole-slide inference: segment one or more class names over a
# slide, gated by Otsu tissue detection.

export CUDA_VISIBLE_DEVICES=0

# Activate the VISTA-PATH environment (see ../environment.yml).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate VISTA-PATH

dataset_name=TCGA-COAD

python3 inference.py \
  --infer_vis_dir ./results/${dataset_name} \
  --checkpoint_file ./checkpoints/pytorch_model.bin \
  --image_file /data/TCGA-COAD/20x_images/TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.tif \
  --class_names "Tumor" "Stroma" \
  --crop_size 1024 \
  --overlap 128 \
  --bbx_random 1
