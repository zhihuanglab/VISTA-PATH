import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from PIL import Image
import numpy as np
from models.backbones import CustomSegmentationModel
import pandas as pd
import json
import random
import cv2
import os
import csv

from skimage.io import imread, imsave
from skimage.filters import threshold_otsu

from argparse import ArgumentParser
from utils import evaluate_segmentation, compute_multi_class_metrics, vis_img, save_prob_maps
import torchvision.transforms.functional as TF

import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder

import matplotlib.pyplot as plt

from get_WSI_foreground import get_foreground_mask

from tqdm import tqdm
import tifffile

import openslide


Image.MAX_IMAGE_PIXELS = None

def set_seed(seed):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs (if using multi-GPU)

    torch.backends.cudnn.deterministic = True   # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     

set_seed(42)



def extract_patches(image, mask, patch_size=224, overlap=56):
    """
    Extract patches of size (patch_size x patch_size) with given overlap.
    If one dimension is smaller than patch_size, keep it whole in that dimension
    and only crop along the other.
    """
    H, W = image.shape[:2]

    # clamp overlap so stride is positive
    overlap = max(0, min(overlap, patch_size - 1))
    stride_y = patch_size - overlap if H >= patch_size else H
    stride_x = patch_size - overlap if W >= patch_size else W

    def build_coords(limit, psize, stride):
        if limit <= psize:
            return [0]
        coords = list(range(0, limit - psize + 1, stride))
        if coords[-1] + psize < limit:
            coords.append(limit - psize)
        return coords

    y_coords = build_coords(H, min(H, patch_size), stride_y)
    x_coords = build_coords(W, min(W, patch_size), stride_x)

    patches, mask_patches, positions = [], [], []

    for y in y_coords:
        for x in x_coords:
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patches.append(image[y:y_end, x:x_end])
            mask_patches.append(mask[y:y_end, x:x_end])
            positions.append((x, y))

    return patches, mask_patches, positions



def generate_gaussian_weight_mask(height, width, sigma_scale=0.125):
   
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xv, yv = np.meshgrid(x, y)

    sigma_x = sigma_scale * 2
    sigma_y = sigma_scale * 2

    gauss = np.exp(-((xv**2) / (2 * sigma_x**2) + (yv**2) / (2 * sigma_y**2)))
    gauss /= gauss.max()  

    return gauss.astype(np.float32)


# Wrapper for HuggingFace Trainer compatibility
class SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, box=None):
        logits, _ = self.model(pixel_values, input_ids, attention_mask, box)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits.detach(), "loss": loss}

def main(args):

    os.makedirs(args.infer_vis_dir, exist_ok=True)

    # Define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Using device:', device)
    base_model_name = "vinid/plip"

    print('class names:', args.class_names)


    # Instantiate model
    core_model = CustomSegmentationModel(base_model_name, args.d_model, args.nhead, args.num_layers, args.bbx_random)
    model = SegWrapper(core_model)
    processor = CLIPProcessor.from_pretrained(base_model_name)


    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()


    try:
        if args.image_file.lower().endswith(('.tiff', '.tif')):
            with tifffile.TiffFile(args.image_file) as tf:
                image = tf.series[0].asarray()
        elif args.image_file.endswith(('.svs', '.ndpi')):
            slide = openslide.open_slide(args.image_file)
            width, height = slide.dimensions
            img = slide.read_region((0, 0), 0, (width, height)).convert("RGB")
            image = np.array(img)
        else:
            image = Image.open(args.image_file).convert("RGB")
            image = np.array(image)
    except Exception as e:
        print(f"Failed to load image: {args.image_file}")
        raise e
    

    print('Starting to obtain foreground mask...')
    mask, thumb, scale, H, W = get_foreground_mask(image, max_dim=4096, min_obj=8000, close_radius=3)
    mask = cv2.resize(mask.astype('uint8'), (W, H), interpolation=cv2.INTER_NEAREST)

    print('Obtaining foreground mask is done!')

    H, W = image.shape[:2]


    patches, mask_patches, positions = extract_patches(image, mask, args.crop_size, args.overlap)

    has_overlap = len(positions) > 1

    prob_map_dict = {}
    count_map_dict = {}
    instance_ids = []

    print('Extracting patches is done, and starting inference...')

    for patch, mask_patch, (x_offset, y_offset) in tqdm(zip(patches, mask_patches, positions), total=len(patches)):

        patch_H, patch_W = patch.shape[:2]

        patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask_patch = cv2.resize(mask_patch, (224, 224), interpolation=cv2.INTER_NEAREST)

        if has_overlap:
            weight_mask = generate_gaussian_weight_mask(height=patch_H, width=patch_W)
        else:
            weight_mask = None  


        unique_idx = np.unique(mask_patch)
        unique_idx = unique_idx[unique_idx > 0]

        if len(unique_idx) == 0:
            continue

        for idx, cls_name in enumerate(args.class_names):
            gt2D = mask_patch
            y_indices, x_indices = np.where(gt2D > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bboxes = np.array([x_min, y_min, x_max, y_max])

            text_template = f"an image of {cls_name}"

            print('inference for class:', cls_name, 'at patch position:', x_offset, y_offset)

            inputs = processor(text=text_template, images=patch, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            pixel_values = inputs['pixel_values'].to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            box = torch.tensor(bboxes).float().unsqueeze(0).to(device)

            outputs = model(pixel_values, input_ids, attention_mask, labels = None, box = box)
            outputs = F.softmax(outputs['logits'], dim=1)
            foreground_prob = outputs[0][1].detach().cpu().numpy()  # (224, 224)

            foreground_prob = cv2.resize(foreground_prob, (patch_W, patch_H), interpolation=cv2.INTER_LINEAR)

            if idx not in prob_map_dict:
                prob_map_dict[idx] = np.zeros((H, W), dtype=np.float32)
                count_map_dict[idx] = np.zeros((H, W), dtype=np.float32)
                instance_ids.append(idx)
            

            if has_overlap:
                prob_map_dict[idx][y_offset:y_offset+patch_H, x_offset:x_offset+patch_W] += foreground_prob * weight_mask
                count_map_dict[idx][y_offset:y_offset+patch_H, x_offset:x_offset+patch_W] += weight_mask
            else:
                prob_map_dict[idx][y_offset:y_offset+patch_H, x_offset:x_offset+patch_W] += foreground_prob 
                count_map_dict[idx][y_offset:y_offset+patch_H, x_offset:x_offset+patch_W] = 1.0

    print('start post-processing...')
    foreground_probs_all = []
    template_all = []
    for idx in instance_ids:
        prob = prob_map_dict[idx]
        count = count_map_dict[idx]
        prob_avg = np.divide(prob, count, out=np.zeros_like(prob), where=count > 0)
        foreground_probs_all.append(prob_avg)

        # text_prompt = f"an image of {args.class_names[idx]}"
        template_all.append(args.class_names[idx])

    merged_mask = np.zeros((H, W), dtype=np.uint8)
    mask_components = []

    # Step 1: Threshold each prob map with Otsu and record area
    for idx, prob_map in zip(instance_ids, foreground_probs_all):
        # try:
        #     thresh = threshold_otsu(prob_map)
        # except ValueError:
        #     thresh = 0.5  # fallback if Otsu fails (e.g. uniform map)
        thresh = 0.3

        binary_mask = (prob_map >= thresh).astype(np.uint8)
        area = binary_mask.sum()
        mask_components.append((idx, binary_mask, area))

    # Step 2: Sort by area ascending (so smaller masks overwrite larger)
    mask_components.sort(key=lambda x: -x[2])

    # Step 3: Merge masks by overwriting
    for idx, bin_mask, _ in mask_components:
        merged_mask[bin_mask == 1] = idx + 1

    mapped_mask = merged_mask  # Final output mask, with correct instance_ids

    print('start visualization')

    # imsave(os.path.join(args.index_map_dir, f'{name.split('.')[0]}.png'), np.uint8(mapped_mask))
    img_name = os.path.splitext(os.path.basename(args.image_file))[0]
    vis_img(image, mapped_mask, foreground_probs_all, template_all, os.path.join(args.infer_vis_dir, f'{img_name}.jpg'))

    save_prob_maps(foreground_probs_all, template_all, save_path=os.path.join(args.infer_vis_dir, f'{img_name}.npz'))
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512) 
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=1000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--infer_vis_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_file", type=str, default="/project/zhihuanglab/Peixian/PathSeg_experiment/DL/PLIP_box_combined/checkpoints/Combined_22_vision/checkpoint-32150/pytorch_model.bin")
    parser.add_argument("--image_file", type=str, default="/data/TCGA-COAD/20x_images/TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.tif")
    parser.add_argument("--bbx_random", type=float, default=1)
    parser.add_argument("--class_names", nargs="+",
                        help="List of class names (exclude background)", default=["Tumor", "Stroma"])

    args = parser.parse_args()
    main(args)
    
