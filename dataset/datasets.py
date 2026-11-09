import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # prevent fork warning

import json
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def dataset_name_from_path(mask_path):
    """Return the dataset a mask belongs to: the directory directly above
    `masks`, e.g. `<root>/BRCA/masks/train/x.png` -> `BRCA`. The class-name
    JSON for that dataset must be `<json_dir>/<dataset>.json`."""
    parts = os.path.normpath(mask_path).split(os.sep)
    if "masks" not in parts:
        raise ValueError(f"Mask path has no 'masks' directory: {mask_path}")
    i = len(parts) - 1 - parts[::-1].index("masks")
    if i == 0:
        raise ValueError(f"No dataset directory above 'masks': {mask_path}")
    return parts[i - 1]


class PathSegDataset(Dataset):
    """Text- and box-prompted binary segmentation samples for VISTA-PATH.

    Each CSV row is one (image, mask, class index) triple. A sample is a random
    `crop_size` crop of the image, resized to `image_size`, together with the
    binary mask of that class, a jittered bounding box around it, and the text
    prompt "an image of <class name>". Crops that contain none of the classes
    listed for their dataset fall back to an all-background target with the
    prompt "an image of background" and an all-zero box.
    """

    def __init__(self, csv_file, json_dir, processor, phase="train",
                 crop_size=(512, 512), bbox_shift=20, augmentations=False,
                 image_size=512, m2f_processor=None):
        self.data = pd.read_csv(csv_file, low_memory=False)

        # PLIP (CLIP) processor, used only to tokenize the text prompts.
        self.processor = processor
        # Mask2Former processor: rescale + ImageNet normalization. do_resize is
        # disabled at construction because crops are resized here.
        self.m2f_processor = m2f_processor
        self.crop_size = crop_size
        self.image_size = image_size
        self.bbox_shift = bbox_shift

        self.img_list = self.data[f"{phase}_img"].dropna().tolist()
        self.mask_list = self.data[f"{phase}_mask"].dropna().tolist()
        self.idx_list = self.data[f"{phase}_idx"].dropna().tolist()

        if augmentations:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.aug = None

        # Load every class-name JSON once.
        self.json_cache = {}
        for fname in os.listdir(json_dir):
            if fname.endswith(".json"):
                with open(os.path.join(json_dir, fname), "r") as f:
                    self.json_cache[os.path.splitext(fname)[0]] = json.load(f)

        missing = sorted({dataset_name_from_path(m) for m in self.mask_list} - set(self.json_cache))
        if missing:
            raise FileNotFoundError(f"No class-name JSON in {json_dir} for datasets: {missing}")

        # Tokenize every prompt once, before the DataLoader forks, and move the
        # tensors to shared memory so all workers share one copy.
        self._build_text_token_cache()

    def _build_text_token_cache(self):
        self.text_token_cache = {}
        all_class_names = {"background"}
        for idx_to_class in self.json_cache.values():
            all_class_names.update(idx_to_class.values())
        for class_name in all_class_names:
            text = f"an image of {class_name}"
            tok = self.processor(text=text, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=77)
            ids = tok["input_ids"].squeeze(0).share_memory_()
            attn = tok["attention_mask"].squeeze(0).share_memory_()
            self.text_token_cache[text] = (ids, attn)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]
        idx_to_class = self.json_cache[dataset_name_from_path(mask_path)]

        image_array = np.array(Image.open(img_path).convert("RGB"))
        mask_array = np.array(Image.open(mask_path))
        assert mask_array.shape[:2] == image_array.shape[:2], \
            f"Image/mask size mismatch: {img_path} vs {mask_path}"

        if mask_array.dtype == bool:
            mask_array = mask_array.astype(np.uint8)
        if mask_array.ndim == 3:
            mask_array = np.uint8(mask_array[:, :, 0] > 0)

        h, w = mask_array.shape
        crop_h = min(h, self.crop_size[0])
        crop_w = min(w, self.crop_size[1])
        S = self.image_size

        # Up to 20 random crops; accept the first one containing any labelled class.
        gt2D = np.zeros((S, S), dtype=np.uint8)
        cropped_image = image_array[:crop_h, :crop_w, :]
        valid_crop = False
        for _ in range(20):
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            cropped_image = image_array[top:top + crop_h, left:left + crop_w, :]
            cropped_mask = mask_array[top:top + crop_h, left:left + crop_w]

            labels = [l for l in np.unique(cropped_mask) if l > 0 and str(l) in idx_to_class]
            if labels:
                cropped_image = cv2.resize(cropped_image, (S, S), interpolation=cv2.INTER_LINEAR)
                cropped_mask = cv2.resize(cropped_mask, (S, S), interpolation=cv2.INTER_NEAREST)
                selected_label = int(self.idx_list[idx])
                gt2D = (cropped_mask == selected_label).astype(np.uint8)
                valid_crop = True
                break

        if not valid_crop:
            cropped_image = cv2.resize(cropped_image, (S, S), interpolation=cv2.INTER_LINEAR)

        if self.aug is not None:
            out = self.aug(image=cropped_image, mask=gt2D)
            cropped_image, gt2D = out["image"], out["mask"]
        image = Image.fromarray(cropped_image)

        if gt2D.max() == 0:
            bboxes = np.array([0, 0, 0, 0])
            text_template = "an image of background"
        else:
            # Tight box around the target class, each side jittered outward.
            y_idx, x_idx = np.where(gt2D > 0)
            H, W = gt2D.shape
            x_min = max(0, x_idx.min() - random.randint(0, self.bbox_shift))
            x_max = min(W, x_idx.max() + random.randint(0, self.bbox_shift))
            y_min = max(0, y_idx.min() - random.randint(0, self.bbox_shift))
            y_max = min(H, y_idx.max() + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
            text_template = f"an image of {idx_to_class[str(selected_label)]}"

        pixel_values_m2f = self.m2f_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        input_ids, attention_mask = self.text_token_cache[text_template]

        return (pixel_values_m2f, input_ids, attention_mask,
                torch.tensor(np.array(gt2D), dtype=torch.long),
                text_template, torch.tensor(bboxes).float())
