"""Build the training CSV that train.py reads.

Expects the layout produced by the released data and the VISTA-PATH_data/
pre-processing scripts:

    <data_root>/<dataset>/images/train/<name>.png
    <data_root>/<dataset>/masks/train/<name>.png

(`images/` and `masks/` without a `train/` sub-folder are accepted as well).
Each dataset needs a class map `<json_dir>/<dataset>.json`.

The CSV has one row per (image, class) pair: a class gets a row when its index
is non-zero, is listed in the dataset's JSON, and covers at least
`--min_pixels` pixels of the mask (30 for the released checkpoint).

    python build_train_csv.py --data_root ./data --json_dir ./idx_to_names --output train.csv
"""
import csv
import json
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def mask_classes(mask_path, class_ids, min_pixels):
    mask = np.array(Image.open(mask_path))
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    if mask.ndim == 3:  # RGB masks are treated as binary, as in training
        mask = np.uint8(mask[:, :, 0] > 0)
    values, counts = np.unique(mask, return_counts=True)
    return [int(v) for v, c in zip(values, counts)
            if v > 0 and int(v) in class_ids and c >= min_pixels]


def _job(item):
    img, mask, class_ids, min_pixels = item
    return img, mask, mask_classes(mask, class_ids, min_pixels)


def collect_pairs(dataset_dir):
    for sub in (os.path.join("masks", "train"), "masks"):
        mask_dir = os.path.join(dataset_dir, sub)
        if os.path.isdir(mask_dir) and any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(mask_dir)):
            img_dir = mask_dir.replace(os.sep + "masks", os.sep + "images", 1)
            break
    else:
        return []
    pairs = []
    for fname in sorted(os.listdir(mask_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        img = os.path.join(img_dir, fname)
        if not os.path.exists(img):
            print(f"  skipping {fname}: no matching image in {img_dir}")
            continue
        pairs.append((img, os.path.join(mask_dir, fname)))
    return pairs


def main(args):
    datasets = args.datasets or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d)))

    jobs = []
    for ds in datasets:
        json_file = os.path.join(args.json_dir, f"{ds}.json")
        if not os.path.exists(json_file):
            print(f"{ds}: no {json_file}, skipped")
            continue
        with open(json_file) as f:
            class_ids = {int(k) for k in json.load(f)}
        pairs = collect_pairs(os.path.join(args.data_root, ds))
        print(f"{ds}: {len(pairs)} image/mask pairs")
        jobs += [(img, mask, class_ids, args.min_pixels) for img, mask in pairs]

    n_rows = 0
    with open(args.output, "w", newline="") as f, ProcessPoolExecutor(args.num_workers) as pool:
        writer = csv.writer(f)
        writer.writerow(["train_img", "train_mask", "train_idx"])
        for img, mask, classes in pool.map(_job, jobs, chunksize=64):
            for c in classes:
                writer.writerow([img, mask, c])
                n_rows += 1
    print(f"Wrote {n_rows} rows to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder holding one sub-folder per dataset.")
    parser.add_argument("--json_dir", type=str, default="./idx_to_names")
    parser.add_argument("--output", type=str, default="train.csv")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only these dataset folders (default: all with a JSON).")
    parser.add_argument("--min_pixels", type=int, default=30,
                        help="Minimum pixels a class must cover in a mask to get a row.")
    parser.add_argument("--num_workers", type=int, default=8)
    main(parser.parse_args())
