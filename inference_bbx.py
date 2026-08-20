"""VISTA-PATH inference with expert-provided box prompts.

Each image is paired with a label mask whose non-zero values are class indices
(the ``--json_file`` mapping). For every sliding window and every class present
in that window, the tight bounding box of the class is handed to the model as a
box prompt, together with the class-name text prompt.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.io import imsave
from transformers import AutoImageProcessor, CLIPProcessor

from inference_utils import (
    BASE_MODEL_NAME,
    MASK2FORMER_NAME,
    add_model_args,
    encode_class_texts,
    generate_gaussian_weight_mask,
    get_patch_positions,
    load_model,
    set_seed,
)
from utils import save_prob_maps, vis_img_bbx

Image.MAX_IMAGE_PIXELS = None

set_seed(42)

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def build_tasks(image, mask, positions, crop_size, S):
    """One task per (window, class present in that window).

    Returns a list of ``(x, y, patch_H, patch_W, class_idx, bbox)`` where bbox is
    the tight box of that class inside the window, expressed in the S x S model
    input coordinates the box prompt encoder expects.
    """
    H, W = image.shape[:2]
    tasks = []

    for (x, y) in positions:
        patch_H = min(crop_size, H - y)
        patch_W = min(crop_size, W - x)
        if patch_H <= 0 or patch_W <= 0:
            continue

        mask_patch = mask[y:y + patch_H, x:x + patch_W]
        # Nearest-neighbour so class indices survive the resize; boxes are then
        # taken in the same S x S frame the model sees.
        mask_S = cv2.resize(mask_patch, (S, S), interpolation=cv2.INTER_NEAREST)

        for c in np.unique(mask_S):
            if c == 0:
                continue
            ys_idx, xs_idx = np.where(mask_S == c)
            if len(xs_idx) == 0:
                continue
            bbox = np.array([
                float(xs_idx.min()), float(ys_idx.min()),
                float(xs_idx.max() + 1), float(ys_idx.max() + 1),
            ], dtype=np.float32)
            tasks.append((x, y, patch_H, patch_W, int(c), bbox))

    return tasks


def main(args):
    os.makedirs(args.infer_vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # bbx_random is the probability of *dropping* the box prompt; this
    # entrypoint is box-prompted, so it defaults to 0 (always keep the box).
    model = load_model(args, device)

    processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME)
    # The patch is resized to m2f_image_size below, so the processor only
    # rescales + ImageNet-normalizes.
    m2f_processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_NAME,
        do_resize=False,
        do_reduce_labels=False,
        ignore_index=255,
    )

    S = args.m2f_image_size

    image_names = sorted(f for f in os.listdir(args.image_dir)
                         if f.lower().endswith(IMG_EXTS))
    mask_names = sorted(f for f in os.listdir(args.mask_dir)
                        if f.lower().endswith(IMG_EXTS))
    assert len(image_names) == len(mask_names), \
        "Number of images and masks must be the same"

    with open(args.json_file, 'r') as f:
        idx_to_class = json.load(f)
    if "0" not in idx_to_class:
        idx_to_class["0"] = "Background"
    # Keys arrive as strings from JSON; the model side indexes by int.
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    # Prompt tokens for every class in the mapping, computed once.
    text_cache = encode_class_texts(processor, idx_to_class, sorted(idx_to_class))

    for name, mask_name in zip(image_names, mask_names):
        img_path = os.path.join(args.image_dir, name)
        mask_path = os.path.join(args.mask_dir, mask_name)
        slide_id = os.path.splitext(name)[0]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        assert mask.shape[:2] == image.shape[:2], \
            f"{name}: image {image.shape[:2]} and mask {mask.shape[:2]} differ in size"

        H, W = image.shape[:2]

        positions = get_patch_positions(H, W, args.crop_size, args.overlap)
        has_overlap = len(positions) > 1 and args.overlap > 0
        tasks = build_tasks(image, mask, positions, args.crop_size, S)

        if not tasks:
            print(f"[{name}] mask is empty, skipping")
            continue

        print(f"[{name}] {len(positions)} windows, {len(tasks)} (window, class) tasks")

        prob_map_dict = {}
        count_map_dict = {}
        weight_mask_cache = {}

        with torch.no_grad():
            for b0 in range(0, len(tasks), args.batch_size):
                batch = tasks[b0:b0 + args.batch_size]

                patches = [
                    cv2.resize(image[y:y + pH, x:x + pW], (S, S),
                               interpolation=cv2.INTER_LINEAR)
                    for x, y, pH, pW, _, _ in batch
                ]
                pixel_values_m2f = m2f_processor(
                    images=patches, return_tensors="pt")["pixel_values"].to(device)

                boxes = torch.tensor(np.stack([t[5] for t in batch]),
                                     dtype=torch.float32, device=device)
                input_ids = torch.stack([text_cache[t[4]][0] for t in batch]).to(device)
                attn_mask = torch.stack([text_cache[t[4]][1] for t in batch]).to(device)

                outputs = model(pixel_values_m2f, input_ids, attn_mask,
                                labels=None, box=boxes)
                probs = F.softmax(outputs['logits'], dim=1)
                fps = probs[:, 1].detach().cpu().numpy()  # (N, S, S)

                for k, (x, y, pH, pW, c, _) in enumerate(batch):
                    fp = cv2.resize(fps[k], (pW, pH), interpolation=cv2.INTER_LINEAR)

                    if c not in prob_map_dict:
                        prob_map_dict[c] = np.zeros((H, W), dtype=np.float32)
                        count_map_dict[c] = np.zeros((H, W), dtype=np.float32)

                    if has_overlap:
                        key = (pH, pW)
                        if key not in weight_mask_cache:
                            weight_mask_cache[key] = generate_gaussian_weight_mask(pH, pW)
                        wm = weight_mask_cache[key]
                        prob_map_dict[c][y:y + pH, x:x + pW] += fp * wm
                        count_map_dict[c][y:y + pH, x:x + pW] += wm
                    else:
                        prob_map_dict[c][y:y + pH, x:x + pW] += fp
                        count_map_dict[c][y:y + pH, x:x + pW] += 1.0

        # ── average the overlaps, then Otsu-threshold each class ────────────
        class_indices = sorted(prob_map_dict)
        foreground_probs_all, gt_masks_all, template_all = [], [], []
        mask_components = []

        # Compact the (possibly sparse) label indices to 1..N for visualization.
        old_to_new = {0: 0}

        for kth, c in enumerate(class_indices):
            prob = prob_map_dict[c]
            count = count_map_dict[c]
            valid = count > 0
            prob[valid] /= count[valid]

            foreground_probs_all.append(prob)
            gt_masks_all.append((mask == c).astype(np.uint8))
            template_all.append(idx_to_class[c])
            old_to_new[c] = kth + 1

            try:
                thresh = threshold_otsu(prob)
            except ValueError:
                thresh = 0.5  # fallback if Otsu fails (e.g. uniform map)
            binary_mask = (prob >= thresh)
            mask_components.append((c, binary_mask, int(binary_mask.sum())))

        # Sort by area descending so smaller masks overwrite larger ones.
        mask_components.sort(key=lambda t: -t[2])

        merged_mask = np.zeros((H, W), dtype=np.uint8)
        for c, binary_mask, _ in mask_components:
            merged_mask[binary_mask] = c

        lut = np.zeros(max(int(mask.max()), max(old_to_new)) + 1, dtype=np.uint8)
        for old, new in old_to_new.items():
            lut[old] = new
        mapped_mask = lut[merged_mask]
        mapped_gt = lut[mask]

        # The .jpg figure is always written; the label mask and the
        # probability maps are opt-in.
        vis_path = os.path.join(args.infer_vis_dir, f'{slide_id}.jpg')
        vis_img_bbx(image, mapped_mask, mapped_gt, foreground_probs_all,
                    gt_masks_all, template_all, vis_path)

        if args.save_mask:
            imsave(os.path.join(args.infer_vis_dir, f'{slide_id}.png'), merged_mask)

        if args.save_prob:
            save_prob_maps(foreground_probs_all, template_all,
                           save_path=os.path.join(args.infer_vis_dir, f'{slide_id}.npz'))

        print(f"Done: {name} -> {vis_path}")

    print("\nInference complete. Results saved to:", args.infer_vis_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="VISTA-PATH inference with box prompts.")
    add_model_args(parser)

    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of images to segment.")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory of label masks providing the box prompts. "
                             "Pixel values are class indices matching --json_file.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="JSON mapping of label index -> class name, e.g. "
                             "./idx_to_names/BRCA.json")
    parser.add_argument("--infer_vis_dir", type=str, default="./results",
                        help="Output directory for the predicted masks.")

    parser.add_argument("--crop_size", type=int, default=1024,
                        help="Sliding-window size in image pixels. Each window is resized "
                             "to --m2f_image_size before being fed to the model.")
    parser.add_argument("--overlap", type=int, default=128,
                        help="Overlap in pixels between adjacent sliding-window patches. "
                             "When > 0 and more than one patch, overlaps are blended with a "
                             "Gaussian weight mask.")
    parser.add_argument("--bbx_random", type=float, default=0,
                        help="Probability of dropping the box prompt. Keep at 0 so every "
                             "window is conditioned on its box.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of (window, class) pairs per GPU forward pass.")
    parser.add_argument("--save_mask", action="store_true",
                        help="Also save the label mask as <image>.png, keeping the label "
                             "indices of the input mask. Off by default; only the "
                             "<image>.jpg figure is written.")
    parser.add_argument("--save_prob", action="store_true",
                        help="If set, save the per-class probability maps as <image>.npz, "
                             "keyed by class name.")

    args = parser.parse_args()
    main(args)
