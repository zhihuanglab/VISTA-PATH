import matplotlib.pyplot as plt
import numpy as np
import cv2

from matplotlib.patches import Rectangle

from sklearn.metrics import confusion_matrix

def quick_resize(img, max_dim=1024, is_mask=False):
    """
    Resize so that the longest edge ≤ max_dim, preserving aspect ratio.
    - is_mask=True → nearest neighbor (保持整数类别)
    - is_mask=False → area interpolation (更平滑，适合图像/概率)
    """
    h, w = img.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1.0:  # 只缩小，不放大
        new_w, new_h = int(w * scale), int(h * scale)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img

def vis_img(image, pred_mask, foreground_probs_all, template_all, save_path):

    img_small = quick_resize(image, max_dim=1024)
    pred_mask_small = quick_resize(pred_mask, max_dim=1024, is_mask=True)
    foreground_probs_small = [quick_resize(p, max_dim=1024) for p in foreground_probs_all]

    num_rows = len(foreground_probs_small) + 1

    # (4) Plotting
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4*num_rows))

    for i in range(num_rows - 1):

        axs[i, 0].imshow(img_small)
        axs[i, 0].set_title(template_all[i])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(foreground_probs_small[i], cmap="gray")
        axs[i, 1].set_title("Predicted Mask")
        axs[i, 1].axis("off")


    n_classes = len(template_all) + 1
    cmap = plt.cm.get_cmap('tab20', n_classes)  # discrete colormap
    axs[num_rows - 1, 0].imshow(img_small)
    # axs[0].set_title(text_prompt)
    axs[num_rows - 1, 0].axis("off")

    im1 = axs[num_rows - 1, 1].imshow(pred_mask_small, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 1].set_title("Predicted Mask")
    axs[num_rows - 1, 1].axis("off")

    cbar = fig.colorbar(im1, ax=axs[num_rows - 1, 1], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels(["background"] +template_all)

    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def vis_img_bbx(image, pred_mask, true_mask, foreground_probs_all, gt_masks_all, template_all, save_path):

    num_rows = len(foreground_probs_all) + 1

    # (4) Plotting
    fig, axs = plt.subplots(len(foreground_probs_all)+1, 3, figsize=(12, 4*num_rows))

    for i in range(num_rows - 1):

        axs[i, 0].imshow(image)
        axs[i, 0].set_title(template_all[i])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(foreground_probs_all[i], cmap="gray")
        axs[i, 1].set_title("Predicted Mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(gt_masks_all[i], cmap="gray")
        axs[i, 2].set_title("Ground Truth Mask")
        axs[i, 2].axis("off")

    n_classes = len(template_all) + 1
    cmap = plt.cm.get_cmap('tab20', n_classes)  # discrete colormap
    cmap = plt.cm.get_cmap('tab20', n_classes)  # discrete colormap

    axs[num_rows - 1, 0].imshow(image)
    # axs[0].set_title(text_prompt)
    axs[num_rows - 1, 0].axis("off")

    axs[num_rows - 1, 1].imshow(pred_mask,  cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 1].set_title("Predicted Mask")
    axs[num_rows - 1, 1].axis("off")


    im1 = axs[num_rows - 1, 2].imshow(true_mask,  cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
    axs[num_rows - 1, 2].set_title("Ground Truth Mask")
    axs[num_rows - 1, 2].axis("off")


    cbar = fig.colorbar(im1, ax=axs[num_rows - 1, 2], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels(["background"] + template_all)

    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

def evaluate_segmentation(pred_mask, true_mask, eps=1e-7):
    """
    Evaluates binary segmentation mask performance using NumPy arrays.

    Args:
        pred_mask (np.ndarray): Predicted binary mask or probability map, shape [H, W] or [1, H, W]
        true_mask (np.ndarray): Ground truth binary mask, same shape as pred_mask
        eps (float): Small epsilon to avoid division by zero

    Returns:
        tuple: (dice, accuracy, precision, recall)
    """

    # Binarize masks (threshold at 0)
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
   
    metrics = {}
    epsilon = 1e-7

    num_classes = max(int(np.amax(gt)), int(np.amax(pred))) + 1

    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=np.arange(num_classes))

    # Per-class metrics
    IoU = []
    Dice = []
    Precision = []
    Recall = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        # Skip classes not present in the GT
        if (TP + FN) == 0:
            continue  # Class i not present in GT, skip it

        iou = TP / (TP + FP + FN + epsilon)
        dice = 2 * TP / (2 * TP + FP + FN + epsilon)
        prec = TP / (TP + FP + epsilon)
        rec = TP / (TP + FN + epsilon)

        IoU.append(iou)
        Dice.append(dice)
        Precision.append(prec)
        Recall.append(rec)

   

    metrics["IoU"] = np.mean(IoU)
    metrics["Dice"] = np.mean(Dice)
    metrics["Precision"] = np.mean(Precision)
    metrics["Recall"] = np.mean(Recall)

    return metrics


def save_prob_maps(foreground_probs_all, template_all, save_path="prob_maps.npz"):
    """
    foreground_probs_all: list of numpy arrays, 每个是一个概率图
    template_all: list of str, 类别名称
    save_path: 输出路径
    """
    assert len(foreground_probs_all) == len(template_all), "数量不一致"

    # 用字典构造
    prob_dict = {name: prob for name, prob in zip(template_all, foreground_probs_all)}

    # 保存为 npz
    np.savez(save_path, **prob_dict)
    print(f"Saved probability maps to {save_path}")
