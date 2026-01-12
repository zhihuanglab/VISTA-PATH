# VISTA-PATH: An interactive foundation model for histopathology image segmentation

## 🔥 Overview

VISTA-PATH (Visual Interactive Segmentation and Tissue Analysis for Pathology) is an interactive foundation model for histopathology image segmentation that integrates visual context, textual class prompts, and expert-guided interaction. Pre-trained on over **1.4 million** samples, VISTA-PATH achieves strong segmentation generalization across organs and tissue types, supports efficient **human-in-the-loop** refinement, and enables **clinically interpretable analysis** through survival-associated morphological features.

## Model Overview

![overview](figures/architecture.png)


## 🚀 Installation

First clone the repo (including the model checkpoint) and cd into the repo

```
git lfs install
git clone https://github.com/zhihuanglab/VISTA-PATH.git
cd VISTA-PATH
git lfs pull
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

## 🧠 Model Download

The VISTA-PATH model can be downloaded from

```
https://github.com/zhihuanglab/VISTA-PATH/tree/main/checkpoints/pytorch_model.bin
```

## 🔬 Quick Start: Model Inference  

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

## 🌟 Acknowledgement

The project was built on top of repositories such as [PLIP](https://github.com/PathologyFoundation/plip) and [SAM](https://github.com/facebookresearch/segment-anything). We thank the authors and developers for their contribution.


## 📖 Publication


