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
  --image_file ./examples/TCGA-COAD/TCGA-AD-6895-01Z-00-DX1.7FB1FBC6-683B-4285-89D1-A7A20F07A9D4.svs \
  --class_names "Tumor" "Stroma" \
  --crop_size 2048 \
  --overlap 256 \
  --bbx_random 1
