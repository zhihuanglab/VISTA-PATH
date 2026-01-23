#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
CODE_DIR=/home/peixian/VISTA-PATH
cd ${CODE_DIR}

source activate PathSeg

dataset_name=TCGA-COAD

python3 inference.py \
  --infer_vis_dir ./results/${dataset_name} \
  --checkpoint_file checkpoints/pytorch_model.bin \
  --image_file ./examples/TCGA-COAD/TCGA-AD-6895-01Z-00-DX1.7FB1FBC6-683B-4285-89D1-A7A20F07A9D4.svs \
  --bbx_random 1 \
  --class_names "Tumor" "Stroma" \
  --overlap 512
