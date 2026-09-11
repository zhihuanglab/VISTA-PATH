import numpy as np

from PIL import Image
from skimage.io import imsave, imread
import os
import cv2

img_dir = './datasets/GlaS/dataset'

img_names = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]

output_img_dir = './data/GlaS/images'
output_mask_dir = './data/GlaS/masks'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

for name in img_names:
    img = Image.open(os.path.join(img_dir, name))
    img = np.array(img)

    if 'ann' in name:
        img = np.uint8(img > 0)
        raw = imread(os.path.join(img_dir, name[:-9]+'.bmp'))
        resized_img = cv2.resize(img, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        imsave(os.path.join(output_mask_dir, name.split('.')[0][:-5]+'.png'), resized_img)
    else:

        if os.path.exists(os.path.join(img_dir, name[:-4]+'_anno.bmp')):
            imsave(os.path.join(output_img_dir, name.split('.')[0]+'.png'), img)

