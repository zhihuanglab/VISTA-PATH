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
from utils import evaluate_segmentation, compute_multi_class_metrics, vis_img_bbx, save_prob_maps
import torchvision.transforms.functional as TF

import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder

import matplotlib.pyplot as plt


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
    # os.makedirs(args.index_map_dir, exist_ok=True)

    # Define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = "vinid/plip"


    # Instantiate model
    core_model = CustomSegmentationModel(base_model_name, args.d_model, args.nhead, args.num_layers, args.bbx_random)
    model = SegWrapper(core_model)
    processor = CLIPProcessor.from_pretrained(base_model_name)


    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()

    image_names = [f for f in os.listdir(args.image_dir) if f.endswith('.png') or f.endswith('.tif')]
    mask_names = [f for f in os.listdir(args.mask_dir) if f.endswith('.png') or f.endswith('.tif')]

    image_names.sort()
    mask_names.sort()

    assert len(image_names) == len(mask_names), "Number of images and masks must be the same"


    with open(args.json_file, 'r') as f:
            idx_to_class = json.load(f)
    
    if "0" not in idx_to_class:
        idx_to_class["0"] = "Background"

    n_classes = len(idx_to_class)

    full_names = [idx_to_class[str(i)] for i in range(n_classes)]

    for name, mask_name in zip(image_names, mask_names):

        img_path = os.path.join(args.image_dir, name)
        mask_path = os.path.join(args.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Failed to load image: {img_path}")
            raise e

        try:
            mask = Image.open(mask_path)
            mask = np.array(mask)
        except Exception as e:
            print(f"Failed to load mask: {mask_path}")
            raise e

        H, W = image.shape[:2]

        
        patches, mask_patches, positions = extract_patches(image, mask, args.crop_size, 128)

        has_overlap = len(positions) > 1

        prob_map_dict = {}
        count_map_dict = {}
        instance_ids = []

        for patch, mask_patch, (x_offset, y_offset) in zip(patches, mask_patches, positions):

            patch_H, patch_W = patch.shape[:2]

            patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_patch, (224, 224), interpolation=cv2.INTER_NEAREST)

            if has_overlap:
                weight_mask = generate_gaussian_weight_mask(height=patch_H, width=patch_W)
            else:
                weight_mask = None  


            unique_idx = np.unique(mask_patch)
            unique_idx = unique_idx[unique_idx > 0]

            for idx in unique_idx:
                gt2D = (mask_patch == idx).astype(np.uint8)
                y_indices, x_indices = np.where(gt2D > 0)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    print('skipping')
                    continue

                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bboxes = np.array([x_min, y_min, x_max, y_max])

                text_template = f"an image of {idx_to_class[str(idx)]}"

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


        foreground_probs_all = []
        gt_masks_all = []
        template_all = []
        old_to_new_mapping = {}
        old_to_new_mapping[0] = 0  # background
        for kth, idx in enumerate(instance_ids):
            prob = prob_map_dict[idx]
            count = count_map_dict[idx]
            prob_avg = np.divide(prob, count, out=np.zeros_like(prob), where=count > 0)
            foreground_probs_all.append(prob_avg)
            
            gt_mask = (mask == idx).astype(np.uint8)  # same shape as full image
            gt_masks_all.append(gt_mask)

            # text_prompt = f"an image of {idx_to_class[str(idx)]}"
            template_all.append(idx_to_class[str(idx)])
            old_to_new_mapping[idx] = kth + 1  # start from 1, 0 is background
        

        ## prepare for the relabeling for visualization
        # e.g. if the instance ids are [3, 5, 10], we map them to [1, 2, 3]
        # so that the color map is consistent
        lut = np.zeros(max(old_to_new_mapping)+1, dtype=np.int32)
        for old, new in old_to_new_mapping.items():
            lut[old] = new

        merged_mask = np.zeros((H, W), dtype=np.uint8)
        mask_components = []

        # Step 1: Threshold each prob map with Otsu and record area
        for idx, prob_map in zip(instance_ids, foreground_probs_all):
            try:
                thresh = threshold_otsu(prob_map)
            except ValueError:
                thresh = 0.5  # fallback if Otsu fails (e.g. uniform map)
            binary_mask = (prob_map >= thresh).astype(np.uint8)
            area = binary_mask.sum()
            mask_components.append((idx, binary_mask, area))

        # Step 2: Sort by area ascending (so smaller masks overwrite larger)
        mask_components.sort(key=lambda x: -x[2])

        # Step 3: Merge masks by overwriting
        for idx, bin_mask, _ in mask_components:
            merged_mask[bin_mask == 1] = idx

        mapped_mask = lut[merged_mask]  # Final output mask, with correct instance_ids

        mask = lut[mask]  # also remap the original mask for fair comparison

        # imsave(os.path.join(args.index_map_dir, f'{name.split('.')[0]}.png'), np.uint8(mapped_mask))
        vis_img_bbx(image, mapped_mask, mask, foreground_probs_all, gt_masks_all, template_all, os.path.join(args.infer_vis_dir, f'{name.split('.')[0]}.jpg'))

        save_prob_maps(foreground_probs_all, template_all, save_path=os.path.join(args.infer_vis_dir, f'{name.split('.')[0]}.npz'))
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512) 
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--json_file", type=str, default="/project/zhihuanglab/Peixian/Path_Seg/data/meta_data/idx_to_names/")
    parser.add_argument("--infer_vis_dir", type=str, default="./results/BRCA_2cls")
    parser.add_argument("--checkpoint_file", type=str, default="./checkpoints/combined_sampled/model_epoch_40.pt")
    parser.add_argument("--image_dir", type=str, default="./vis_images/BRCA_2cls")
    parser.add_argument("--mask_dir", type=str, default="./vis_images/BRCA_2cls")
    parser.add_argument("--bbx_random", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
    
