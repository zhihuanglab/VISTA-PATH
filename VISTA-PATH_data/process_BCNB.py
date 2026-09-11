import os
import openslide
from PIL import Image

import shutil

img_dir = './datasets/BCNB/WSIs'
output_dir = './data/BCNB/images'

os.makedirs(output_dir, exist_ok = True)


img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

for name in img_names:

    shutil.copy(os.path.join(img_dir, name), os.path.join(output_dir, name))

    
    