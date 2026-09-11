import os
import numpy as np
import shutil

from skimage.io import imread, imsave

data_dirs = ['./datasets/ORCA/test', './datasets/ORCA/validation']

img_output_dir = './data/ORCA/images'
mask_output_dir = './data/ORCA/masks'

os.makedirs(img_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)


for sub_dir in data_dirs:
    img_names = [f for f in os.listdir(sub_dir) if f.endswith('.png')]

    for name in img_names:
        if 'mask' in name:
            mask = imread(os.path.join(sub_dir, name))
            mask = mask[:,:,0]
            modified_mask = mask.copy()
            modified_mask[(modified_mask < 100)] = 0
            modified_mask[(modified_mask >= 100) & (modified_mask <130)] = 1
            modified_mask[(modified_mask >= 130)] = 2
            imsave(os.path.join(mask_output_dir, name[:-9]+'.png'), modified_mask)
        else:
            shutil.copy(os.path.join(sub_dir, name), os.path.join(img_output_dir, name))

