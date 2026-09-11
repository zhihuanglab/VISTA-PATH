import os
import openslide
from PIL import Image
from skimage.io import imread
import shutil

Image.MAX_IMAGE_PIXELS = None

img_dir = './data/KMAP/images'
output_dir = './data/KMAP/images'

mask_dir = './data/KMAP/masks'

os.makedirs(output_dir, exist_ok = True)


img_names = [f for f in os.listdir(img_dir) if f.endswith('.svs')]

for name in img_names:


    if not os.path.exists(os.path.join(mask_dir, name.split('.')[0]+'.png')):
        print('not exist!')
        print(os.path.join(mask_dir, name.split('.')[0]+'.png'))

    # slide = openslide.open_slide(os.path.join(img_dir, name))
    # width, height = slide.dimensions

    # mask = imread(os.path.join(mask_dir, name.split('.')[0]+'.png'))
    # assert mask.shape[1] == width
    # assert mask.shape[0] == height

    # img = slide.read_region((0, 0), 0, (width, height))
    # img.save(os.path.join(output_dir, name.split('.')[0]+'.ome.png'))

    # shutil.copy(os.path.join(img_dir, name), os.path.join(output_dir, name))
    
    