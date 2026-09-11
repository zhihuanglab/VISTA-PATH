import os
import pandas as pd
import random
from glob import glob
import shutil

# Directories for images and masks
mask_dir = [
    './datasets/KPIs/train/Task1_patch_level/train', \
    './datasets/KPIs/val/Task1_patch_level/validation', \
    './datasets/KPIs/test/Task1_patch_level/test'
]

img_result_dir = "./data/KPIs/images"
mask_result_dir = "./data/KPIs/masks"

os.makedirs(img_result_dir, exist_ok=True)
os.makedirs(mask_result_dir, exist_ok=True)


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

output_file = "./data/meta_data/train_test_split/KPIs.csv"

# Collect all image and mask files
img_files = []
mask_files = []

for sub_dir in mask_dir:
    sub_folders = [f for f in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, f))]

    for sub_folder in sub_folders:
        slide_ids = [f for f in os.listdir(os.path.join(sub_dir, sub_folder)) if os.path.isdir(os.path.join(sub_dir, sub_folder, f))]

        for slide_id in slide_ids:
            mask_files.extend([os.path.join(sub_dir, sub_folder, slide_id, 'mask', f) for f in os.listdir(os.path.join(sub_dir, sub_folder, slide_id, 'mask')) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])
            img_files.extend([os.path.join(sub_dir, sub_folder, slide_id, 'img', f) for f in os.listdir(os.path.join(sub_dir, sub_folder, slide_id, 'img')) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])


# Ensure images and masks are matched
img_files.sort()
mask_files.sort()
assert len(img_files) == len(mask_files), "Mismatch between image and mask files."

for img_name, mask_name in zip(img_files, mask_files):

    if 'train' in img_name:
        prefix = 'train'
    elif 'test' in img_name:
        prefix = 'test'
    elif 'val' in img_name:
        prefix = 'val'
    
    save_name = prefix + '_' + os.path.basename(img_name).split('.')[0]+'.png'

    shutil.copy(img_name, os.path.join(img_result_dir, save_name))
    shutil.copy(mask_name, os.path.join(mask_result_dir, save_name))
