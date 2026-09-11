import os
import openslide
from PIL import Image

img_dir = './data/BACH/image'
output_dir = './data/BACH/image'

os.makedirs(output_dir, exist_ok = True)


img_names = [f for f in os.listdir(img_dir) if f.endswith('.svs')]

for name in img_names:

    slide = openslide.open_slide(os.path.join(img_dir, name))
    width, height = slide.dimensions
    print(slide.dimensions)
    
    img = slide.read_region((0, 0), 0, (width, height))
    img.save(os.path.join(output_dir, name.split('.')[0]+'.png'))
    