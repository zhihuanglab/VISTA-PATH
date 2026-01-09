#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CODE_DIR=/home/peixian/PASeg/
cd ${CODE_DIR}

source activate PathSeg

dataset_name=TCGA-COAD

# python3 inference.py \
#   --infer_vis_dir ./results/${dataset_name} \
#   --checkpoint_file checkpoints/pytorch_model.bin \
#   --image_file /data/TCGA-COAD/20x_images/TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.tif \
#   --bbx_random 1 \
#   --class_names "Tumor" "Stroma" \
#   --overlap 1000

python3 inference.py \
  --infer_vis_dir ./results/3D \
  --checkpoint_file checkpoints/pytorch_model.bin \
  --image_file /home/peixian/PASeg/3D/virtual_HE_z00002.tif \
  --bbx_random 1 \
  --class_names "Glands" \
  --overlap 256