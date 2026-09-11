import os
import openslide
from PIL import Image
import numpy as np
from skimage.io import imread, imsave

import shutil
import csv 

def remap_mask(mask, provider):

    print(provider)
    if provider == "radboud":
        new_mask = np.zeros_like(mask)
        new_mask[(mask == 1) | (mask == 2)] = 1
        new_mask[(mask == 3) | (mask == 4) | (mask == 5)] = 2

        print('merged!')
        return new_mask
    else:
        return mask  

data = {}


for filename in ['./datasets/PANDA/train.csv', './datasets/PANDA/test.csv']:
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['image_id']] = row['data_provider']

img_dir = './datasets/PANDA/train_images'
output_dir = './data/PANDA/images'

mask_dir = './datasets/PANDA/train_label_masks'
mask_output_dir = './data/PANDA/masks'

os.makedirs(output_dir, exist_ok = True)
os.makedirs(mask_output_dir, exist_ok = True)

img_names = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]

for name in img_names:

    if os.path.exists(os.path.join(mask_dir,  name.split('.')[0]+'_mask.tiff')):

        if os.path.exists(os.path.join(mask_output_dir, name[:-10]+'.png')):
            continue
        
        try:
            mask = imread(os.path.join(mask_dir,  name.split('.')[0]+'_mask.tiff'))
        except Exception as e:
            print(f"Skipping: {e}")
            continue 

        mask = mask[:,:,0]

        mask = remap_mask(mask, data[name.split('.')[0]])
        imsave(os.path.join(mask_output_dir, name.split('.')[0]+'.png'), mask)
        # shutil.copy(os.path.join(mask_dir,  name.split('.')[0]+'_mask.tiff'), os.path.join(mask_output_dir, name[:-10]+'.png'))

        shutil.copy(os.path.join(img_dir, name), os.path.join(output_dir, name.split('.')[0]+'.png'))

    
    