# VISTA-PATH: An interactive foundation model for histopathology image segmentation

## 📖 Overview

VISTA-PATH (Visual Interactive Segmentation and Tissue Analysis for Pathology) is an interactive foundation model for histopathology image segmentation that integrates visual context, textual class prompts, and expert-guided interaction. Pre-trained on over **1.4 million** samples, VISTA-PATH achieves strong segmentation generalization across organs and tissue types, supports efficient **human-in-the-loop** refinement, and enables **clinically interpretable analysis** through survival-associated morphological features.


## Installation

First clone the repo and cd into the repo

```
git clone https://github.com/lilab-stanford/MUSK](https://github.com/zhihuanglab/VISTA-PATH.git
cd VISTA-PATH
```

Create a new enviroment with anaconda.

```
conda create -n VISTA-PATH python=3.12
conda activate VISTA-PATH

conda install -c conda-forge scikit-image opencv pandas pillow numpy

conda install -c conda-forge openslide openslide-python

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers==4.46.1

pip install pycocotools matplotlib scikit-learn

pip install accelerate==0.26.0

conda install -c conda-forge opencv

conda install -c conda-forge albumentations
```

## Model Download

The VISTA-PATH model can be downloaded from

```
https://github.com/zhihuanglab/VISTA-PATH/tree/main/checkpoints
```

## Quick Start: Model Inference  

put the checkpoint into the file ./checkpoints

### a. do inference without bbx

```
dataset_name=TCGA-COAD

python3 inference.py \
  --infer_vis_dir ./results/${dataset_name} \
  --checkpoint_file ./checkpoints/pytorch_model.bin \
  --image_file /data/TCGA-COAD/20x_images/TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.tif \
  --bbx_random 1 \
  --class_names "Tumor" "Stroma"

```

`--bbx_random` indicates to use bbx prompts or not, `1` means not using bbx, `0` means using bbx 

`--infer_vis_dir` saves final outputs. Two types of files are save: .jpg shows visual example, `.npz` saves probability maps produced by the model for each class, which in the format: [class_name]: [probability map]


### b. do inference with bbx

Toy examples are provided in `./examples/BRCA` for quick start

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

`--mask_dir` provides bbx prompts

`--json_file` provides class names


## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the VISTA-PATH model and its derivatives, which include models trained on outputs from the VISTA-PATH model or datasets created from the VISTA-PATH model, is prohibited and requires prior approval.
