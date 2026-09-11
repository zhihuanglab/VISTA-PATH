import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image
import re

import openslide
import shutil

from collections import defaultdict
import matplotlib.pyplot as plt


def vis_images(image, mask, save_path):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='jet')
    plt.title("Mask")
    plt.axis("off")

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)  # customize dpi if needed
    plt.close()

def extract_single_label(label_txt):
    return label_txt.split('|')[0].strip()

def parse_annotation_file(annotation_path):
    annotations = []
    labels = set()

    # Regex pattern for: filename,type,result,name,data
    pattern = r'^([^,]+),([^,]+),([^,]+),(.*?),({.*})'

    with open(annotation_path, "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header

            match = re.match(pattern, line)
            if not match:
                print(f"⚠️  Skipping malformed line {i}: {line[:80]}...")
                continue

            file_name = match.group(1).strip()
            label_txt = match.group(4).strip()
            label = extract_single_label(label_txt)
            labels.add(label)

            data_str = match.group(5).strip()
            try:
                data = json.loads(data_str.replace("'", '"'))  # try to fix bad quotes
            except Exception as e:
                print(f"⚠️  Skipping JSON parse error at line {i}: {e}")
                continue

            annotations.append((file_name, label, data))

    return annotations, labels

def find_files_dict(root_dir, extensions=('.svs', '.ndpi')):
    file_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                full_path = os.path.join(dirpath, fname)
                file_dict[fname] = full_path
    return file_dict

invalid_classes = ['nan', '*', '**', '??']

# === CONFIGURATION ===
csv_path = './datasets/DROID/drsk/annotations.csv'  # Your annotation CSV file
image_dir = './datasets/DROID/drsk/'              # Folder containing the .png files (converted from .ndpi)

file_dict = find_files_dict(image_dir)


output_mask_dir = './data/DROID/masks/'         # Where to save output masks
os.makedirs(output_mask_dir, exist_ok=True)

output_img_dir = './data/DROID/images/'         # Where to save output masks
os.makedirs(output_img_dir, exist_ok=True)

vis_dir = './data/DROID/vis/'         # Where to save output masks
os.makedirs(vis_dir, exist_ok=True)

# === LOAD CSV ===
annotations, labels = parse_annotation_file(csv_path)

# === Get consistent class-to-ID mapping globally ===
all_class_names = [n for n in labels if n not in invalid_classes]
class_to_id = {name: i + 1 for i, name in enumerate(all_class_names)}  # Start from 1 (0 = background)

id_to_class_str = {str(v): k for k, v in class_to_id.items()}

# Save to JSON
with open("./stat/meta_data/idx_to_names/DROID.json", "w") as f:
    json.dump(id_to_class_str, f, indent=4)

print("Class-to-ID mapping:")
for k, v in class_to_id.items():
    print(f"{v}: {k}")


# === GROUP ANNOTATIONS BY IMAGE ===
image_to_annotations = defaultdict(list)
for file_name, label, data in annotations:
    if label in class_to_id:
        image_to_annotations[file_name].append((class_to_id[label], data))


# === RENDER MASKS ===
for image_name, anno_list in image_to_annotations.items():
    image_path = file_dict[image_name]
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        continue

    print(f"🖼️  Processing: {image_name}")
    slide = openslide.OpenSlide(image_path)  # or .svs

    # Read a region (x, y, level, size)
    region = slide.read_region(location=(0, 0), level=0, size=slide.dimensions)
    image = region.convert("RGB")
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)

    for class_id, data in anno_list:
        for poly in data.get("polygons", []):
            pts = poly.get("p", [])
            if not pts:
                continue
            pts_pixel = np.array([[int(p['x'] * width), int(p['y'] * width)] for p in pts], dtype=np.int32)
            cv2.fillPoly(mask, [pts_pixel], color=class_id)

    # Save mask
    mask_name = os.path.splitext(image_name)[0] + ".png"
    mask_path = os.path.join(output_mask_dir, mask_name)
    Image.fromarray(mask).save(mask_path)
    print(f"✅ Saved: {mask_path}")
    shutil.copy(image_path, os.path.join(output_img_dir, image_name))

    vis_images(image, mask, os.path.join(vis_dir, mask_name))

