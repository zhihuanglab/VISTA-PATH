# pip install tifffile opencv-python scikit-image numpy
import numpy as np
import tifffile
import cv2
from skimage import morphology, filters
from skimage.io import imsave
import os

import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_img(img, mask, save_path=None):

    # ---- 绘图 ----
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def read_tiff_thumbnail(img, max_dim=4096):

    Ht, Wt = img.shape[:2]

    if max(Ht, Wt) > max_dim:
        scale = max(Ht, Wt) / float(max_dim)
    else:
        scale = 1.0
    img = cv2.resize(img, (int(Wt / scale), int(Ht / scale)), interpolation=cv2.INTER_AREA)

    return img, scale, Ht, Wt

def tissue_mask_lab(img_rgb, min_obj=5000, close_radius=3):
    """
    LAB 的 ab 能量 + Otsu，形态学清理，输出 uint8 {0,1}
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[...,1] - 128.0
    b = lab[...,2] - 128.0
    ab = np.sqrt(a*a + b*b)
    ab_u8 = cv2.normalize(ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thr = filters.threshold_otsu(ab_u8)
    mask = (ab_u8 > thr)

    # 形态学：闭运算 → 填洞/去小块
    if close_radius > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_radius+1, 2*close_radius+1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)

    mask = morphology.remove_small_holes(mask, area_threshold=min_obj)
    mask = morphology.remove_small_objects(mask, min_size=min_obj)
    return mask.astype(np.uint8)

def get_foreground_mask(image, max_dim=4096, min_obj=5000, close_radius=3):
    thumb, scale, H, W = read_tiff_thumbnail(image, max_dim=max_dim)
    mask = tissue_mask_lab(thumb, min_obj=min_obj, close_radius=close_radius)
    return mask, thumb, scale, H, W

