import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import openslide
from skimage.io import imread, imsave

import shutil

Image.MAX_IMAGE_PIXELS = None

mask_dir = "./datasets/prostate_gland_segmnetation/patient_024147_slide_01/mask"
img_dir = "./datasets/prostate_gland_segmnetation/patient_024147_slide_01/img"

result_mask_dir = "./data/prostate_gland_segmentation/masks"
result_img_dir = "./data/prostate_gland_segmentation/images"

os.makedirs(result_mask_dir, exist_ok = True)
os.makedirs(result_img_dir, exist_ok = True)

mask_names = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]

for name, mask_n in zip(img_names, mask_names):

    mask = imread(os.path.join(mask_dir, mask_n))
    mask = np.uint8(mask > 0)

    imsave(os.path.join(result_mask_dir, mask_n[:-9]+'.png'), mask)
    shutil.copy(os.path.join(img_dir, name), os.path.join(result_img_dir, name))