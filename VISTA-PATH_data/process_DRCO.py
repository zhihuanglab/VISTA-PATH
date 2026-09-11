import pandas as pd
import json
import os
import cv2
import numpy as np
from PIL import Image
import openslide
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import re
import csv

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

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

def extract_single_label(label_txt):
    return label_txt.split('|')[0].strip()

def parse_annotation_csv(csv_path):
    annotations = []
    labels = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header = lines[0]

        for i, line in enumerate(lines[1:], start=2):
            try:
                parts = line.strip().split(",")

                # Find where the JSON starts
                json_start_idx = None
                for j, part in enumerate(parts):
                    if part.strip().startswith('{'):
                        json_start_idx = j
                        break

                if json_start_idx is None:
                    print(f"⚠️ Line {i}: JSON start not found, skipping")
                    continue

                file_name = parts[0].strip()
                label_txt = parts[3].strip()
                label = extract_single_label(label_txt)
                labels.add(label)

                # Join the remaining parts as JSON string
                data_str = ",".join(parts[json_start_idx:]).strip()
                data_str = data_str.replace("'", '"')
                data_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', data_str)

                data = json.loads(data_str)
                annotations.append((file_name, label, data))

            except Exception as e:
                print(f"⚠️ Skipping line {i} due to error: {e}")
                continue

    return annotations, labels



def find_files_dict(root_dir, extensions=('.svs', '.ndpi', '.png', '.tif', '.jpg')):
    file_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                full_path = os.path.join(dirpath, fname)
                file_dict[fname] = full_path
    return file_dict

# === CONFIGURATION ===
csv_path = './datasets/DRCO/annotations.csv'
image_dir = './datasets/DRCO/'

output_mask_dir = './data/DRCO/masks/'
os.makedirs(output_mask_dir, exist_ok=True)

output_img_dir = './data/DRCO/images/'
os.makedirs(output_img_dir, exist_ok=True)

vis_dir = './data/DRCO/vis/'
os.makedirs(vis_dir, exist_ok=True)

# === Load Annotations ===
annotations, labels = parse_annotation_csv(csv_path)

print(labels)

# === Create Class-to-ID Mapping ===
invalid_classes = ['nan', '*', '**', '??']
all_class_names = [n for n in labels if n not in invalid_classes]
class_to_id = {name: i + 1 for i, name in enumerate(sorted(all_class_names))}
id_to_class_str = {str(v): k for k, v in class_to_id.items()}

with open("./stat/meta_data/idx_to_names/DRCO.json", "w") as f:
    json.dump(id_to_class_str, f, indent=4)

print("Class-to-ID mapping:")
for k, v in class_to_id.items():
    print(f"{v}: {k}")

# === Map Images to Annotations ===
image_to_annotations = defaultdict(list)
for file_name, label, data in annotations:
    if label in class_to_id:
        image_to_annotations[file_name].append((class_to_id[label], data))

file_dict = find_files_dict(image_dir)

# === Generate Masks ===
for image_name, anno_list in image_to_annotations.items():
    image_path = file_dict.get(image_name)
    if not image_path or not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_name}")
        continue

    print(f"🖼️ Processing: {image_name}")
    slide = openslide.OpenSlide(image_path)
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

    mask_name = os.path.splitext(image_name)[0] + ".png"
    mask_path = os.path.join(output_mask_dir, mask_name)
    Image.fromarray(mask).save(mask_path)
    print(f"✅ Saved: {mask_path}")

    shutil.copy(image_path, os.path.join(output_img_dir, image_name))
    vis_images(image, mask, os.path.join(vis_dir, mask_name))
