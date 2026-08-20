import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, AutoImageProcessor
from PIL import Image
import numpy as np
import cv2
import gc

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

from skimage.io import imread, imsave
from skimage.filters import threshold_otsu

from argparse import ArgumentParser
import torch.nn.functional as F

import openslide

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None

set_seed(42)

# Extensions read lazily through OpenSlide (pyramidal WSIs). Everything else is
# loaded fully into RAM via skimage.imread (small PNG/JPEG ROIs).
WSI_EXTS = ('.svs', '.ndpi', '.mrxs', '.tif', '.tiff')


def generate_foreground_mask(image, blur_ksize=7, morph_ksize=15):
    """Tissue/foreground detection via Otsu on the HSV saturation channel.

    Bright slide background has low saturation; stained tissue is highly
    saturated. Returns a binary (H, W) uint8 mask where 1 == foreground/tissue.

    For WSIs this is run on a downsampled thumbnail (see get_foreground_mask),
    never on the level-0 image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # Light blur to suppress speckle before thresholding.
    saturation = cv2.GaussianBlur(saturation, (blur_ksize, blur_ksize), 0)

    try:
        thresh = threshold_otsu(saturation)
    except ValueError:
        thresh = 0  # uniform image -> everything is foreground

    fg = (saturation > thresh).astype(np.uint8)

    # Close small holes, then drop small speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    return fg


def get_foreground_mask(img_path, full_image_array, is_wsi, H, W, thumb_max):
    """
    Compute the tissue/foreground mask on a downsampled thumbnail and return
    ``(fg_small, sx, sy)`` where ``fg_small`` is a (h_s, w_s) uint8 mask and
    ``(sx, sy)`` are the level-0 -> thumbnail downsample factors
    (``thumb_x = level0_x / sx``).

    The level-0 image is never materialized: for WSIs the thumbnail comes from
    OpenSlide's ``get_thumbnail``; for in-RAM arrays it is a cv2 downsample.
    Otsu tissue detection (HSV saturation) runs on this small thumbnail, which
    is both fast and memory-safe for arbitrarily large slides.
    """
    if is_wsi:
        slide = openslide.open_slide(img_path)
        thumb = slide.get_thumbnail((thumb_max, thumb_max))
        slide.close()
        thumb = np.array(thumb)[:, :, :3]
    else:
        scale = min(1.0, float(thumb_max) / max(H, W))
        tw = max(1, int(round(W * scale)))
        th = max(1, int(round(H * scale)))
        thumb = cv2.resize(full_image_array[:, :, :3], (tw, th),
                           interpolation=cv2.INTER_AREA)

    fg_small = generate_foreground_mask(thumb)
    h_s, w_s = fg_small.shape[:2]
    sx = W / float(w_s)
    sy = H / float(h_s)
    return fg_small, sx, sy


def build_tasks_otsu(fg_small, sx, sy, positions, crop_size, S, H, W,
                     class_indices, fg_thresh):
    """
    Build the per-(patch, class) task list from the thumbnail tissue mask.

    Each task is one (patch, class) pair:
        (x, y, pH, pW, class_idx, bbox)
    For every sliding window we look up the matching region of the thumbnail
    tissue mask. Windows with less than ``fg_thresh`` tissue are skipped. For
    the surviving windows the tissue bounding box (in S x S model-input
    coordinates) becomes the bbox prompt, and one task is emitted per class
    (every class is run on every tissue patch, mirroring the original
    per-patch class loop). The full-resolution tissue mask is never built.
    """
    h_s, w_s = fg_small.shape[:2]
    tasks = []

    for (x, y) in positions:
        patch_H = min(crop_size, H - y)
        patch_W = min(crop_size, W - x)
        if patch_H <= 0 or patch_W <= 0:
            continue

        # Map this level-0 window onto the thumbnail tissue mask.
        tx0 = max(0, min(int(np.floor(x / sx)), w_s))
        tx1 = max(0, min(int(np.ceil((x + patch_W) / sx)), w_s))
        ty0 = max(0, min(int(np.floor(y / sy)), h_s))
        ty1 = max(0, min(int(np.ceil((y + patch_H) / sy)), h_s))
        if tx1 <= tx0 or ty1 <= ty0:
            continue

        fg_sub = fg_small[ty0:ty1, tx0:tx1]
        # Skip windows that contain (almost) no tissue.
        if fg_sub.mean() < fg_thresh:
            continue

        # Bbox prompt = tight tissue bounding box in S x S coords. Resizing the
        # thumbnail sub-mask to S (nearest) then taking its extent matches the
        # original "scale full-res fg bbox by S/patch" computation.
        fg_S = cv2.resize(fg_sub, (S, S), interpolation=cv2.INTER_NEAREST)
        ys_idx, xs_idx = np.where(fg_S > 0)
        if len(xs_idx) == 0:
            continue
        bbox = np.array([
            float(xs_idx.min()), float(ys_idx.min()),
            float(xs_idx.max() + 1), float(ys_idx.max() + 1),
        ], dtype=np.float32)

        for c in class_indices:
            tasks.append((x, y, patch_H, patch_W, int(c), bbox))

    return tasks


def get_fg_roi(fg_small, sx, sy, y0, y1, x0, x1):
    """
    Upscale the thumbnail tissue mask to full resolution within the class ROI
    ``[y0:y1, x0:x1]`` (level-0 coords). Returns a (y1-y0, x1-x0) uint8 mask
    used to zero out probabilities outside tissue before Otsu, exactly as the
    original ``prob_avg[fg_mask == 0] = 0`` step did — but only over the tight
    ROI instead of the whole slide.
    """
    h_s, w_s = fg_small.shape[:2]
    tx0 = max(0, min(int(np.floor(x0 / sx)), w_s))
    tx1 = max(0, min(int(np.ceil(x1 / sx)), w_s))
    ty0 = max(0, min(int(np.floor(y0 / sy)), h_s))
    ty1 = max(0, min(int(np.ceil(y1 / sy)), h_s))
    tx1 = max(tx1, tx0 + 1)
    ty1 = max(ty1, ty0 + 1)
    fg_sub = fg_small[ty0:ty1, tx0:tx1]
    return cv2.resize(fg_sub, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def save_thumbnail_vis(img_path, full_image_array, is_wsi, merged_mask,
                       save_path, max_size=2048):
    """
    Save a side-by-side thumbnail figure: raw image (left) vs segmentation
    mask (right). The WSI is never loaded at full resolution — for WSIs we use
    OpenSlide's get_thumbnail; otherwise the in-memory array is downsampled.
    """
    H, W = merged_mask.shape[:2]

    # Raw-image thumbnail (RGB), keeping aspect ratio within max_size.
    if is_wsi:
        slide = openslide.open_slide(img_path)
        thumb = slide.get_thumbnail((max_size, max_size))
        slide.close()
        raw_thumb = np.array(thumb)[:, :, :3]
    else:
        scale = min(1.0, float(max_size) / max(H, W))
        tw = max(1, int(round(W * scale)))
        th = max(1, int(round(H * scale)))
        raw_thumb = cv2.resize(full_image_array[:, :, :3], (tw, th), interpolation=cv2.INTER_AREA)

    # Match the mask thumbnail to the raw thumbnail's size (nearest = label-safe).
    th_h, th_w = raw_thumb.shape[:2]
    mask_thumb = cv2.resize(merged_mask, (th_w, th_h), interpolation=cv2.INTER_NEAREST)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].imshow(raw_thumb)
    axs[0].set_title("Raw Image")
    axs[0].axis("off")

    vmax = max(1, int(merged_mask.max()))
    # Qualitative colormap + nearest interpolation so discrete class labels render
    # as flat, distinct colors instead of being blended into a gradient by
    # matplotlib's default antialiased resampling at draw time.
    axs[1].imshow(mask_thumb, cmap="tab20", vmin=0, vmax=vmax, interpolation="nearest")
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


class WSIROIDataset(Dataset):
    """
    Lazily reads each window patch (per-worker OpenSlide handle, fork-safe) and
    runs the Mask2Former image preprocessing on it. Text features are looked up
    from a precomputed per-class cache, so workers never touch the CLIP encoder.

    A tiny single-entry region cache avoids re-reading the same window when it
    is run for multiple classes (tasks are sorted by (x, y) so same-window
    tasks are consecutive).
    """

    def __init__(self, img_path, full_image_array, tasks, m2f_processor, text_cache, S):
        self.tasks = tasks
        self.img_path = img_path                  # WSI path, None for in-RAM array
        self.full_image_array = full_image_array  # numpy array when not a WSI
        self.m2f_processor = m2f_processor
        self.text_cache = text_cache              # class_idx -> (input_ids, attention_mask)
        self.S = S
        self._slide = None                        # lazy, per-worker handle
        self._cache_key = None
        self._cache_patch = None

    def _open_slide(self):
        if self._slide is None and self.img_path is not None:
            self._slide = openslide.open_slide(self.img_path)

    def _read_region(self, x, y, pH, pW):
        key = (x, y, pH, pW)
        if key == self._cache_key:
            return self._cache_patch

        if self.img_path is not None:
            self._open_slide()
            region = self._slide.read_region((x, y), 0, (pW, pH))
            patch = np.array(region)[:, :, :3]
        else:
            patch = self.full_image_array[y:y+pH, x:x+pW]

        self._cache_key = key
        self._cache_patch = patch
        return patch

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        x, y, pH, pW, class_idx, bboxes = self.tasks[i]

        patch = self._read_region(x, y, pH, pW)

        # Resize to the Mask2Former input resolution; the processor only
        # rescales + ImageNet-normalizes (do_resize=False set in main()).
        patch = cv2.resize(patch, (self.S, self.S), interpolation=cv2.INTER_LINEAR)
        pixel_values_m2f = self.m2f_processor(images=patch, return_tensors="pt")["pixel_values"].squeeze(0)

        input_ids, attention_mask = self.text_cache[class_idx]

        return (
            pixel_values_m2f,                            # (3, S, S)
            torch.tensor(bboxes, dtype=torch.float32),   # (4,)
            input_ids,                                   # (77,)
            attention_mask,                              # (77,)
            x, y, pH, pW, class_idx,
        )


def main(args):

    os.makedirs(args.infer_vis_dir, exist_ok=True)

    # Define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # VISTA-PATH: PLIP text encoder + Mask2Former (Swin) trunk + SAM box prompt
    # encoder. The released checkpoint loads strictly into this architecture.
    model = load_model(args, device)

    processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME)
    # Mask2Former image processor: ImageNet normalization. The patch is already
    # resized to m2f_image_size in the dataset, so do_resize is disabled here.
    m2f_processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_NAME,
        do_resize=False,
        do_reduce_labels=False,
        ignore_index=255,
    )

    S = args.m2f_image_size

    # Either a single slide (--image_file) or every slide in a directory
    # (--image_dir). --image_file wins when both are given.
    if args.image_file:
        image_dir = os.path.dirname(os.path.abspath(args.image_file))
        image_names = [os.path.basename(args.image_file)]
    elif args.image_dir:
        image_dir = args.image_dir
        image_names = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(WSI_EXTS) or f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_names.sort()
        image_names = image_names[args.start_index:args.end_index]
    else:
        raise SystemExit("Provide either --image_file or --image_dir")

    # Class names come from the command line (e.g. --class_names tumor stroma),
    # not from a JSON mapping. Indices are assigned sequentially starting at 1
    # (0 is reserved for background/unlabeled).
    class_entries = [(i + 1, name) for i, name in enumerate(args.class_names)]
    idx_to_class = {idx: name for idx, name in class_entries}
    class_indices = [idx for idx, _ in class_entries]

    # Per-class PLIP text features (computed once, shared across all slides).
    text_cache = encode_class_texts(processor, idx_to_class, class_indices)

    use_pin = device.type == 'cuda'

    for image_name in image_names:

        slide_id = os.path.splitext(image_name)[0]
        vis_path = os.path.join(args.infer_vis_dir, f'{slide_id}.jpg')
        out_path = os.path.join(args.infer_vis_dir, f'{slide_id}.png')

        # Skip slides that have already been segmented.
        if os.path.exists(vis_path):
            print(f"[WSI] {image_name}: results exist, skipping")
            continue

        img_path = os.path.join(image_dir, image_name)
        is_wsi = image_name.lower().endswith(WSI_EXTS)

        # ── slide dimensions (no whole-slide load for WSIs) ─────────────
        if is_wsi:
            _slide_check = openslide.open_slide(img_path)
            W, H = _slide_check.level_dimensions[0]
            _slide_check.close()
            full_image_array = None
        else:
            full_image_array = imread(img_path)
            if full_image_array.ndim == 2:
                full_image_array = np.stack([full_image_array] * 3, axis=-1)
            full_image_array = full_image_array[:, :, :3]
            H, W = full_image_array.shape[:2]

        # ── Otsu tissue mask on a thumbnail (memory-safe for huge WSIs) ──
        fg_small, sx, sy = get_foreground_mask(
            img_path, full_image_array, is_wsi, H, W, args.fg_thumb_max)

        # ── sliding-window positions over the slide ─────────────────────
        positions = get_patch_positions(H, W, args.crop_size, args.overlap)
        has_overlap = len(positions) > 1 and args.overlap > 0
        print(f"[WSI] {image_name}: {len(positions)} sliding-window positions "
              f"(crop={args.crop_size}, overlap={args.overlap})")

        # ── build (patch, class) tasks gated by the tissue mask ─────────
        tasks = build_tasks_otsu(
            fg_small, sx, sy, positions, args.crop_size, S, H, W,
            class_indices, args.fg_thresh)

        if not tasks:
            print(f"[WSI] {image_name}: no tissue found, writing empty result")
            empty = np.zeros((H, W), dtype=np.uint8)
            save_thumbnail_vis(img_path, full_image_array, is_wsi, empty, vis_path)
            if args.save_mask:
                imsave(out_path, empty)
            del empty
            if full_image_array is not None:
                del full_image_array
            del fg_small
            gc.collect()
            continue

        # Sort by (x, y) so multi-class tasks on the same window are consecutive
        # (lets the dataset's single-entry region cache avoid re-reads).
        tasks.sort(key=lambda t: (t[0], t[1]))

        # ── per-class tight ROI maps ────────────────────────────────────
        # Only the bounding region actually covered by a class's patches is
        # allocated, instead of a full H×W map per class.
        class_bbox = {}   # class_idx -> [y0, y1, x0, x1] in full-image coords
        for x, y, pH, pW, c, _ in tasks:
            if c not in class_bbox:
                class_bbox[c] = [y, y + pH, x, x + pW]
            else:
                b = class_bbox[c]
                b[0] = min(b[0], y)
                b[1] = max(b[1], y + pH)
                b[2] = min(b[2], x)
                b[3] = max(b[3], x + pW)

        prob_map_dict = {}
        count_map_dict = {}
        for c, (y0, y1, x0, x1) in class_bbox.items():
            prob_map_dict[c] = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            count_map_dict[c] = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)

        # Gaussian weight masks (keyed by patch size) for blending overlaps.
        weight_mask_cache = {}

        # ── batched inference ───────────────────────────────────────────
        img_path_for_loader = img_path if is_wsi else None
        dataset = WSIROIDataset(img_path_for_loader, full_image_array, tasks,
                                m2f_processor, text_cache, S)

        # Few tasks → worker startup cost outweighs the benefit; fall back to 0.
        effective_workers = args.num_workers if len(tasks) >= args.batch_size * 4 else 0
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=effective_workers,
            pin_memory=use_pin and effective_workers > 0,
            prefetch_factor=2 if effective_workers > 0 else None,
            persistent_workers=False,
            multiprocessing_context='spawn' if effective_workers > 0 else None,
        )

        with torch.no_grad():
            for pixel_values_m2f, bboxes, input_ids, attn_mask, xs, ys, pHs, pWs, class_idxs in dataloader:
                pixel_values_m2f = pixel_values_m2f.to(device, non_blocking=use_pin)
                bboxes = bboxes.to(device, non_blocking=use_pin)
                input_ids = input_ids.to(device, non_blocking=use_pin)
                attn_mask = attn_mask.to(device, non_blocking=use_pin)

                outputs = model(pixel_values_m2f, input_ids, attn_mask, labels=None, box=bboxes)
                probs = F.softmax(outputs['logits'], dim=1)
                fps = probs[:, 1].detach().cpu().numpy()  # (N, S, S)

                del pixel_values_m2f, bboxes, input_ids, attn_mask, outputs, probs

                N = fps.shape[0]
                for k in range(N):
                    x_k = xs[k].item()
                    y_k = ys[k].item()
                    pH_k = pHs[k].item()
                    pW_k = pWs[k].item()
                    c_k = class_idxs[k].item()

                    fp = cv2.resize(fps[k], (pW_k, pH_k), interpolation=cv2.INTER_LINEAR)

                    y0, _, x0, _ = class_bbox[c_k]
                    yr = y_k - y0
                    xr = x_k - x0
                    if has_overlap:
                        key = (pH_k, pW_k)
                        if key not in weight_mask_cache:
                            weight_mask_cache[key] = generate_gaussian_weight_mask(pH_k, pW_k)
                        wm = weight_mask_cache[key]
                        prob_map_dict[c_k][yr:yr+pH_k, xr:xr+pW_k] += fp * wm
                        count_map_dict[c_k][yr:yr+pH_k, xr:xr+pW_k] += wm
                    else:
                        prob_map_dict[c_k][yr:yr+pH_k, xr:xr+pW_k] += fp
                        count_map_dict[c_k][yr:yr+pH_k, xr:xr+pW_k] += 1.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del tasks, dataset, dataloader, weight_mask_cache

        # ── Otsu per class + area-sorted merge into the full-image mask ──
        # Each class's prob map is averaged, zeroed outside tissue (matching the
        # original prob_avg[fg_mask == 0] = 0 step), thresholded with Otsu, then
        # merged smallest-area-last so small masks overwrite large ones.
        mask_components = []
        prob_npz = {}
        for c in list(prob_map_dict.keys()):
            prob = prob_map_dict[c]
            count = count_map_dict[c]
            valid = count > 0
            prob[valid] /= count[valid]
            del count_map_dict[c]

            # Zero out probabilities outside tissue before Otsu.
            y0, y1, x0, x1 = class_bbox[c]
            fg_roi = get_fg_roi(fg_small, sx, sy, y0, y1, x0, x1)
            prob[fg_roi == 0] = 0
            del fg_roi

            if args.save_prob:
                # Store the map over the region this class actually covers,
                # plus its level-0 offsets, instead of a full H x W array.
                cls_name = idx_to_class[c]
                prob_npz[cls_name] = prob.astype(np.float16)
                prob_npz[f"{cls_name}_bbox"] = np.array([y0, y1, x0, x1], dtype=np.int64)

            try:
                thresh = threshold_otsu(prob[::2, ::2])  # Otsu on a downsampled map
            except ValueError:
                thresh = 0.5  # fallback if Otsu fails (e.g. uniform map)
            binary_mask = (prob >= thresh)
            area = int(binary_mask.sum())
            mask_components.append((c, binary_mask, area))
            del prob_map_dict[c]

        # Sort by area descending (so smaller masks overwrite larger ones)
        mask_components.sort(key=lambda x: -x[2])

        merged_mask = np.zeros((H, W), dtype=np.uint8)
        for c, binary_mask, _ in mask_components:
            y0, y1, x0, x1 = class_bbox[c]
            region = merged_mask[y0:y1, x0:x1]  # view into merged_mask
            region[binary_mask] = c

        # The .jpg overview is always written; the label mask and the
        # probability maps are opt-in.
        save_thumbnail_vis(img_path, full_image_array, is_wsi, merged_mask, vis_path)

        if args.save_mask:
            imsave(out_path, merged_mask)

        if args.save_prob:
            npz_path = os.path.join(args.infer_vis_dir, f'{slide_id}.npz')
            np.savez_compressed(npz_path, **prob_npz)
            print(f"Saved probability maps to {npz_path}")
        del prob_npz

        print(f"Done: {image_name} -> {vis_path}")

        del merged_mask, mask_components, prob_map_dict, count_map_dict, fg_small
        if full_image_array is not None:
            del full_image_array
        gc.collect()

    print("\nInference complete. Results saved to:", args.infer_vis_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="VISTA-PATH whole-slide / ROI inference.")
    add_model_args(parser)

    parser.add_argument("--image_file", type=str, default=None,
                        help="Single slide or image to segment (.svs/.ndpi/.mrxs/.tif/.tiff "
                             "read lazily through OpenSlide, .png/.jpg loaded into RAM).")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of slides/images to segment. Ignored when "
                             "--image_file is given.")
    parser.add_argument("--infer_vis_dir", type=str, default="./results",
                        help="Output directory for the predicted masks.")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="One or more class names to segment, e.g. --class_names Tumor Stroma. "
                             "Label indices are assigned in order starting at 1 (0 = background).")

    parser.add_argument("--crop_size", type=int, default=1024,
                        help="Sliding-window size in level-0 pixels. Each window is resized "
                             "to --m2f_image_size before being fed to the model.")
    parser.add_argument("--overlap", type=int, default=128,
                        help="Overlap in pixels between adjacent sliding-window patches. "
                             "When > 0 and more than one patch, overlaps are blended with a "
                             "Gaussian weight mask.")
    parser.add_argument("--bbx_random", type=float, default=1,
                        help="Probability of dropping the box prompt. 1 (default) runs "
                             "prompt-free; 0 conditions every window on the tissue "
                             "bounding box found by Otsu.")

    parser.add_argument("--fg_thresh", type=float, default=0.05,
                        help="Minimum fraction of tissue pixels a sliding window must contain "
                             "to be run through the model. Windows below this are skipped.")
    parser.add_argument("--fg_thumb_max", type=int, default=4096,
                        help="Max dimension of the thumbnail used for Otsu tissue detection. "
                             "Larger gives finer tissue/bbox localization at more memory cost.")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None,
                        help="Process --image_dir entries in [start_index, end_index).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of patches per GPU forward pass")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for parallel patch loading")
    parser.add_argument("--save_mask", action="store_true",
                        help="Also save the label mask as <slide>.png, with pixel values "
                             "equal to the --class_names indices. Off by default; only "
                             "the <slide>.jpg overview is written.")
    parser.add_argument("--save_prob", action="store_true",
                        help="If set, also save the per-class probability maps as "
                             "<slide>.npz. Each class stores its map over the region it "
                             "covers plus a '<class>_bbox' array of [y0, y1, x0, x1] "
                             "level-0 coordinates, so huge slides stay memory-safe.")

    args = parser.parse_args()
    main(args)
