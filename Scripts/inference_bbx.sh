#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CODE_DIR=/home/peixian/PASeg/
cd ${CODE_DIR}

source activate PathSeg

dataset_name=BRCA

python3 inference_bbx.py \
  --infer_vis_dir ./results/${dataset_name} \
  --json_file ./idx_to_names/${dataset_name}.json \
  --checkpoint_file ./checkpoints/pytorch_model.bin \
  --image_dir ./examples/${dataset_name}/images \
  --mask_dir ./examples/${dataset_name}/masks \
  --bbx_random 0 
