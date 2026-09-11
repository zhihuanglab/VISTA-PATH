import os
import openslide
from PIL import Image

import shutil

img_dir = './datasets/hubmap-hacking-the-human-vasculature/images'
output_dir = './data/HuBMAP/images'

mask_dir = './data/HuBMAP/masks'

os.makedirs(output_dir, exist_ok = True)


img_names = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

for name in img_names:

    if not os.path.exists(os.path.join(mask_dir, name.split('.')[0]+'.png')):

        continue

    shutil.copy(os.path.join(img_dir, name), os.path.join(output_dir, name.split('.')[0]+'.png'))

    
    