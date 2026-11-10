"""Convert Visium HD spatial clusters into a segmentation mask.

Takes the 8 um bin k-means clustering that Space Ranger produces for the 10x
Genomics Visium HD Human Lung Cancer (FFPE) sample and paints every bin's
cluster onto a label mask the size of the full-resolution H&E image. Each bin
is drawn as a square around its centre; the background is 0 and clusters keep
their Space Ranger ids (1..k).

Expected input, as downloaded from 10x Genomics, under ./datasets/Visium_Lung/:

    Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf
    binned_outputs/square_008um/spatial/tissue_positions.parquet
    binned_outputs/square_008um/analysis/clustering/
        gene_expression_kmeans_<k>_clusters/clusters.csv

Output: ./data/Visium_Lung/masks/HE_img.png

Reading the parquet file needs pyarrow (pip install pyarrow).
"""
import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from skimage.io import imsave

Image.MAX_IMAGE_PIXELS = None

root_dir = './datasets/Visium_Lung'
image_path = os.path.join(root_dir, 'Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf')
coord_path = os.path.join(root_dir, 'binned_outputs/square_008um/spatial/tissue_positions.parquet')

num_clusters = 4  # which k-means result to use
cluster_path = os.path.join(
    root_dir, 'binned_outputs/square_008um/analysis/clustering',
    f'gene_expression_kmeans_{num_clusters}_clusters', 'clusters.csv')

output_mask_dir = './data/Visium_Lung/masks'
os.makedirs(output_mask_dir, exist_ok=True)


def estimate_bin_size(x, y):
    """Bin side in pixels: the median nearest-neighbour distance between bin
    centres, plus 2 px so neighbouring squares overlap and leave no gaps."""
    coords = np.vstack([x, y]).T
    dists, _ = cKDTree(coords).query(coords, k=2)  # k=2: the point itself, then its neighbour
    return int(round(np.median(dists[:, 1]))) + 2


def generate_bin_mask(x, y, labels, shape, bin_size):
    """Draw each bin as a bin_size square filled with its cluster id."""
    mask = np.zeros(shape, dtype=np.uint8)
    half = bin_size // 2
    for xi, yi, label in zip(x, y, labels):
        x0 = max(int(xi) - half, 0)
        x1 = min(int(xi) + half, shape[1])
        y0 = max(int(yi) - half, 0)
        y1 = min(int(yi) + half, shape[0])
        mask[y0:y1, x0:x1] = int(label)
    return mask


img = mpimg.imread(image_path)
H, W = img.shape[:2]
print(f'H&E image: {H} x {W}')

# Join bin positions with their cluster assignment on the barcode.
coords_df = pd.read_parquet(coord_path)
clusters_df = pd.read_csv(cluster_path).rename(columns={'Barcode': 'barcode'})
merged = coords_df.merge(clusters_df, on='barcode')

x = merged['pxl_col_in_fullres']
y = merged['pxl_row_in_fullres']
cluster_labels = merged['Cluster']

# Drop bins whose centre falls outside the image.
in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
x, y, cluster_labels = x[in_bounds], y[in_bounds], cluster_labels[in_bounds]

bin_size = estimate_bin_size(x, y)
print(f'Estimated bin size: {bin_size} pixels')

mask = generate_bin_mask(x, y, cluster_labels, (H, W), bin_size)
imsave(os.path.join(output_mask_dir, 'HE_img.png'), mask, check_contrast=False)
