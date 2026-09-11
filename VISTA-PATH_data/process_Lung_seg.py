import os
import pandas as pd
import random
from glob import glob
import shutil

# Directories for images and masks
img_dirs = [
    "./datasets/Lung_seg/image",
]
mask_dirs = [
    "./datasets/Lung_seg/mask",
]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

output_dir_img = "./data/Lung_seg/images"
output_dir_mask = "./data/Lung_seg/masks"

os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

# Collect all image and mask files
img_files = []
mask_files = []

for img_dir, mask_dir in zip(img_dirs, mask_dirs):
    img_files.extend(sorted(glob(os.path.join(img_dir, "*.jpg"))))  # Adjust extension if needed
    mask_files.extend(sorted(glob(os.path.join(mask_dir, "*.png"))))

# Ensure images and masks are matched
img_files.sort()
mask_files.sort()
assert len(img_files) == len(mask_files), "Mismatch between image and mask files."

for k, (img, mask) in enumerate(zip(img_files, mask_files)):
    base_name = os.path.basename(mask)
    shutil.copy(img, os.path.join(output_dir_img, base_name))
    shutil.copy(mask, os.path.join(output_dir_mask, base_name))




