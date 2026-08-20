#!/bin/bash
# Box-prompted inference: every class present in a window is prompted with its
# bounding box taken from the mask in --mask_dir.

export CUDA_VISIBLE_DEVICES=0

# Activate the VISTA-PATH environment (see ../environment.yml).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate VISTA-PATH

dataset_name=BRCA

python3 inference_bbx.py \
  --infer_vis_dir ./results/${dataset_name} \
  --json_file ./idx_to_names/${dataset_name}.json \
  --checkpoint_file ./checkpoints/pytorch_model.bin \
  --image_dir ./examples/${dataset_name}/images \
  --mask_dir ./examples/${dataset_name}/masks \
  --bbx_random 0
