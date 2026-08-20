"""Visualization and evaluation helpers used by the inference entrypoints."""

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def get_cmap(name, n):
    """Discrete colormap, compatible with matplotlib >= 3.9.

    ``plt.cm.get_cmap`` was removed in 3.9; ``plt.get_cmap(...).resampled(n)``
    is the supported spelling, with a fallback for older releases.
    """
    cmap = plt.get_cmap(name)
    return cmap.resampled(n) if hasattr(cmap, "resampled") else plt.cm.get_cmap(name, n)


def quick_resize(img, max_dim=1024, is_mask=False):
    """Shrink so the longest edge is <= max_dim, preserving aspect ratio.

    Masks use nearest-neighbour so class indices survive; images/probability
    maps use area interpolation.
    """
    h, w = img.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:  # only shrink, never upscale
        new_w, new_h = int(w * scale), int(h * scale)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img


def vis_img(image, pred_mask, foreground_probs_all, template_all, save_path, max_dim=1024):
    """One row per class (image | probability map) plus a final merged-mask row."""
    img_small = quick_resize(image, max_dim=max_dim)
    pred_mask_small = quick_resize(pred_mask, max_dim=max_dim, is_mask=True)
    probs_small = [quick_resize(p, max_dim=max_dim) for p in foreground_probs_all]

    num_rows = len(probs_small) + 1
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), squeeze=False)

    for i in range(num_rows - 1):
        axs[i, 0].imshow(img_small)
        axs[i, 0].set_title(template_all[i])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(probs_small[i], cmap="gray", vmin=0, vmax=1)
        axs[i, 1].set_title("Predicted Probability")
        axs[i, 1].axis("off")

    n_classes = len(template_all) + 1
    cmap = get_cmap("tab20", n_classes)

    axs[num_rows - 1, 0].imshow(img_small)
    axs[num_rows - 1, 0].set_title("Raw Image")
    axs[num_rows - 1, 0].axis("off")

    im1 = axs[num_rows - 1, 1].imshow(pred_mask_small, cmap=cmap, vmin=0,
                                      vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 1].set_title("Predicted Mask")
    axs[num_rows - 1, 1].axis("off")

    cbar = fig.colorbar(im1, ax=axs[num_rows - 1, 1], orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels(["background"] + list(template_all))

    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def vis_img_bbx(image, pred_mask, true_mask, foreground_probs_all, gt_masks_all,
                template_all, save_path, max_dim=1024):
    """One row per class (image | prediction | ground truth) plus a merged row."""
    img_small = quick_resize(image, max_dim=max_dim)
    pred_mask_small = quick_resize(pred_mask, max_dim=max_dim, is_mask=True)
    true_mask_small = quick_resize(true_mask, max_dim=max_dim, is_mask=True)
    probs_small = [quick_resize(p, max_dim=max_dim) for p in foreground_probs_all]
    gts_small = [quick_resize(g, max_dim=max_dim, is_mask=True) for g in gt_masks_all]

    num_rows = len(probs_small) + 1
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows), squeeze=False)

    for i in range(num_rows - 1):
        axs[i, 0].imshow(img_small)
        axs[i, 0].set_title(template_all[i])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(probs_small[i], cmap="gray", vmin=0, vmax=1)
        axs[i, 1].set_title("Predicted Probability")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(gts_small[i], cmap="gray", vmin=0, vmax=1)
        axs[i, 2].set_title("Ground Truth Mask")
        axs[i, 2].axis("off")

    n_classes = len(template_all) + 1
    cmap = get_cmap("tab20", n_classes)

    axs[num_rows - 1, 0].imshow(img_small)
    axs[num_rows - 1, 0].set_title("Raw Image")
    axs[num_rows - 1, 0].axis("off")

    axs[num_rows - 1, 1].imshow(pred_mask_small, cmap=cmap, vmin=0,
                                vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 1].set_title("Predicted Mask")
    axs[num_rows - 1, 1].axis("off")

    im1 = axs[num_rows - 1, 2].imshow(true_mask_small, cmap=cmap, vmin=0,
                                      vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 2].set_title("Ground Truth Mask")
    axs[num_rows - 1, 2].axis("off")

    cbar = fig.colorbar(im1, ax=axs[num_rows - 1, 2], orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels(["background"] + list(template_all))

    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_prob_maps(foreground_probs_all, template_all, save_path="prob_maps.npz"):
    """Save one probability map per class into a single .npz keyed by class name."""
    assert len(foreground_probs_all) == len(template_all), \
        "probability maps and class names must be the same length"

    prob_dict = {name: prob.astype(np.float16)
                 for name, prob in zip(template_all, foreground_probs_all)}
    np.savez_compressed(save_path, **prob_dict)
    print(f"Saved probability maps to {save_path}")


def evaluate_segmentation(pred_mask, true_mask, eps=1e-7):
    """Binary segmentation metrics: (dice, accuracy, precision, recall)."""
    pred_bin = (pred_mask > 0).astype(np.float32).reshape(-1)
    true_bin = (true_mask > 0).astype(np.float32).reshape(-1)

    intersection = np.sum(pred_bin * true_bin)
    union = np.sum(pred_bin) + np.sum(true_bin)

    dice = (2.0 * intersection + eps) / (union + eps)
    acc = np.mean(pred_bin == true_bin)

    tp = np.sum(pred_bin * true_bin)
    fp = np.sum(pred_bin * (1.0 - true_bin))
    fn = np.sum((1.0 - pred_bin) * true_bin)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return dice, acc, precision, recall


def compute_multi_class_metrics(gt, pred):
    """Macro-averaged IoU / Dice / precision / recall over the classes present in gt."""
    metrics = {}
    epsilon = 1e-7

    num_classes = max(int(np.amax(gt)), int(np.amax(pred))) + 1
    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=np.arange(num_classes))

    IoU, Dice, Precision, Recall = [], [], [], []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP

        # Skip classes not present in the GT
        if (TP + FN) == 0:
            continue

        IoU.append(TP / (TP + FP + FN + epsilon))
        Dice.append(2 * TP / (2 * TP + FP + FN + epsilon))
        Precision.append(TP / (TP + FP + epsilon))
        Recall.append(TP / (TP + FN + epsilon))

    metrics["IoU"] = np.mean(IoU)
    metrics["Dice"] = np.mean(Dice)
    metrics["Precision"] = np.mean(Precision)
    metrics["Recall"] = np.mean(Recall)

    return metrics
