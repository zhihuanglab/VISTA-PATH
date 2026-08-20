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
    encode_class_prompts,
    generate_gaussian_weight_mask,
    get_patch_positions,
    limit_cpu_threads,
    load_model,
    nearest_upsample,
    set_seed,
    tune_for_inference,
    worker_init,
)

from skimage.io import imread
from skimage.filters import threshold_otsu

from argparse import ArgumentParser

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
                     fg_thresh, need_bbox=True):
    """
    Build the per-window task list from the thumbnail tissue mask.

    Each task is one sliding window: ``(x, y, pH, pW, bbox)``. Windows with
    less than ``fg_thresh`` tissue are skipped; for the survivors the tissue
    bounding box (in S x S model-input coordinates) becomes the box prompt.
    The full-resolution tissue mask is never built.

    One task covers *all* classes: the image half of the network is a pure
    function of the pixels, so the window is read and encoded once and only
    the prompt-conditioned decoder is re-run per class (see ``main``). When
    the run is prompt-free (``need_bbox=False``) the per-window box search is
    skipped as well, since the model discards the box anyway.
    """
    h_s, w_s = fg_small.shape[:2]
    tasks = []
    zero_bbox = np.zeros(4, dtype=np.float32)

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

        if not need_bbox:
            tasks.append((x, y, patch_H, patch_W, zero_bbox))
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

        tasks.append((x, y, patch_H, patch_W, bbox))

    return tasks


def get_fg_roi(fg_small, sx, sy, y0, y1, x0, x1, out_h, out_w):
    """
    Upscale the thumbnail tissue mask to the ``out_h x out_w`` accumulator grid
    covering the level-0 region ``[y0:y1, x0:x1]``. Returns a uint8 mask used
    to zero out probabilities outside tissue before Otsu, exactly as the
    original ``prob_avg[fg_mask == 0] = 0`` step did — but only over the tight
    ROI, and at the accumulator's resolution rather than level 0.
    """
    h_s, w_s = fg_small.shape[:2]
    tx0 = max(0, min(int(np.floor(x0 / sx)), w_s))
    tx1 = max(0, min(int(np.ceil(x1 / sx)), w_s))
    ty0 = max(0, min(int(np.floor(y0 / sy)), h_s))
    ty1 = max(0, min(int(np.ceil(y1 / sy)), h_s))
    tx1 = max(tx1, tx0 + 1)
    ty1 = max(ty1, ty0 + 1)
    fg_sub = fg_small[ty0:ty1, tx0:tx1]
    return nearest_upsample(fg_sub, out_h, out_w)


def save_thumbnail_vis(img_path, full_image_array, is_wsi, merged_mask, H, W,
                       save_path, max_size=2048):
    """
    Save a side-by-side thumbnail figure: raw image (left) vs segmentation
    mask (right). Neither side is ever materialized at level 0 — for WSIs the
    raw thumbnail comes from OpenSlide's get_thumbnail, and ``merged_mask`` is
    already the reduced-resolution label map covering the (H, W) slide.
    """
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


def save_label_png(path, mask, compression=1):
    """Write a single-channel label PNG.

    A level-0 whole-slide label map is hundreds of megapixels, and PIL's
    default compression level spends minutes on it for a marginal size win —
    label maps are large flat regions, so the cheapest zlib setting gets
    almost the same file. cv2 also avoids PIL's decompression-bomb ceiling.
    """
    ok = cv2.imwrite(path, mask, [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])
    if not ok:
        raise IOError(f"failed to write {path}")


def pick_read_level(slide, crop_size, S):
    """Pick the pyramid level to read sliding windows from.

    A window is downsampled from ``crop_size`` to ``S`` before it reaches the
    model, so any pyramid level whose downsample is at most ``crop_size / S``
    carries every pixel the model will ever see — and decoding it costs
    ``downsample^2`` times less JPEG than level 0. Returns ``(level,
    downsample)``; falls back to level 0 when no such level exists.
    """
    target = crop_size / float(S)
    if target < 1.0:
        return 0, 1.0
    level = slide.get_best_level_for_downsample(target)
    ds = slide.level_downsamples[level]
    # get_best_level_for_downsample can round up past the target; only accept a
    # level that does not throw away resolution the model would have used.
    if ds > target * 1.001:
        return 0, 1.0
    return level, ds


class WSIROIDataset(Dataset):
    """
    Lazily reads each window (per-worker OpenSlide handle, fork-safe) and
    returns it as a raw uint8 ``(S, S, 3)`` tensor.

    Two things deliberately do *not* happen here. ImageNet normalization is
    left to the GPU: shipping uint8 instead of fp32 cuts the worker->parent
    IPC by 4x and the normalize itself is free on-device. And the window is
    emitted once for all classes rather than once per class, because the
    Mask2Former trunk that consumes it does not depend on the prompt.
    """

    def __init__(self, img_path, full_image_array, tasks, S,
                 read_level=0, read_downsample=1.0):
        self.tasks = tasks
        self.img_path = img_path                  # WSI path, None for in-RAM array
        self.full_image_array = full_image_array  # numpy array when not a WSI
        self.S = S
        self.read_level = read_level
        self.read_downsample = read_downsample
        self._slide = None                        # lazy, per-worker handle

    def _open_slide(self):
        if self._slide is None and self.img_path is not None:
            self._slide = openslide.open_slide(self.img_path)

    def _read_region(self, x, y, pH, pW):
        if self.img_path is not None:
            self._open_slide()
            ds = self.read_downsample
            rw = max(1, int(round(pW / ds)))
            rh = max(1, int(round(pH / ds)))
            region = self._slide.read_region((x, y), self.read_level, (rw, rh))
            return np.asarray(region.convert("RGB"))
        return self.full_image_array[y:y + pH, x:x + pW]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        x, y, pH, pW, bboxes = self.tasks[i]

        patch = self._read_region(x, y, pH, pW)
        if patch.shape[0] != self.S or patch.shape[1] != self.S:
            patch = cv2.resize(patch, (self.S, self.S), interpolation=cv2.INTER_LINEAR)
        patch = np.ascontiguousarray(patch)

        return (
            torch.from_numpy(patch),                     # (S, S, 3) uint8
            torch.from_numpy(bboxes),                    # (4,)
            x, y, pH, pW,
        )


def main(args):

    os.makedirs(args.infer_vis_dir, exist_ok=True)

    # Define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Forward-only knobs: cuDNN autotuning, TF32/AMP, and a cap on the CPU
    # thread pools. Returns the autocast context for the forward pass.
    autocast_ctx = tune_for_inference(args.precision, args.cpu_threads)

    # VISTA-PATH: PLIP text encoder + Mask2Former (Swin) trunk + SAM box prompt
    # encoder. The released checkpoint loads strictly into this architecture.
    model = load_model(args, device)
    core = model.model  # unwrapped, for the split encode/decode path

    processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME)
    # Mask2Former image processor: ImageNet normalization. Only its constants
    # are used — the normalize itself runs on the GPU, on the uint8 patches.
    m2f_processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_NAME,
        do_resize=False,
        do_reduce_labels=False,
        ignore_index=255,
    )
    norm_mean = torch.tensor(m2f_processor.image_mean, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(m2f_processor.image_std, device=device).view(1, 3, 1, 1)
    rescale = float(m2f_processor.rescale_factor)

    S = args.m2f_image_size

    # Accumulator downsample. The model emits an S x S map for a crop_size
    # window, so accumulating at level 0 upsamples predictions only to average
    # and threshold them and then throw the extra pixels away. Working on the
    # model's own grid costs (crop_size/S)^2 less memory and arithmetic, and
    # the label map is upsampled once at the very end. --prob_scale 1 restores
    # level-0 accumulation.
    prob_scale = args.prob_scale if args.prob_scale > 0 else max(1, args.crop_size // S)

    def rc(p):
        """level-0 coordinate -> accumulator-grid coordinate."""
        return int(round(p / prob_scale))

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

    # Per-class PLIP text tokens, encoded once for the whole run and kept on
    # the GPU: the text tower depends only on the class name. Kept in fp32 —
    # it is a handful of 77-token sequences, and the decoder's autocast casts
    # them on use.
    prompt_cache = encode_class_prompts(
        core, processor, idx_to_class, class_indices, device)

    # bbx_random is the probability of *dropping* the box prompt; at 1 the
    # model never looks at it, so the tissue bbox is not worth computing.
    use_box = args.bbx_random < 1.0

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
        read_level, read_ds = 0, 1.0
        if is_wsi:
            _slide_check = openslide.open_slide(img_path)
            W, H = _slide_check.level_dimensions[0]
            if args.read_level == "auto":
                read_level, read_ds = pick_read_level(_slide_check, args.crop_size, S)
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
              f"(crop={args.crop_size}, overlap={args.overlap}, "
              f"read level={read_level}, accum 1/{prob_scale})")

        # ── build the window list, gated by the tissue mask ─────────────
        tasks = build_tasks_otsu(
            fg_small, sx, sy, positions, args.crop_size, S, H, W,
            args.fg_thresh, need_bbox=use_box)

        RH_full, RW_full = rc(H), rc(W)

        if not tasks:
            print(f"[WSI] {image_name}: no tissue found, writing empty result")
            empty = np.zeros((RH_full, RW_full), dtype=np.uint8)
            save_thumbnail_vis(img_path, full_image_array, is_wsi, empty, H, W, vis_path)
            if args.save_mask:
                save_label_png(out_path, np.zeros((H, W), dtype=np.uint8),
                               args.png_compression)
            del empty
            if full_image_array is not None:
                del full_image_array
            del fg_small
            gc.collect()
            continue

        print(f"[WSI] {image_name}: {len(tasks)} tissue windows "
              f"x {len(class_indices)} classes")

        # ── one tight ROI, shared by every class ────────────────────────
        # Every class is evaluated on the same window set, so they share both
        # the bounding region and the overlap-count map.
        y0 = min(t[1] for t in tasks)
        y1 = max(t[1] + t[2] for t in tasks)
        x0 = min(t[0] for t in tasks)
        x1 = max(t[0] + t[3] for t in tasks)
        RY0, RY1, RX0, RX1 = rc(y0), rc(y1), rc(x0), rc(x1)
        RH, RW = max(1, RY1 - RY0), max(1, RX1 - RX0)

        # Accumulator-grid placement of every window, precomputed once.
        placements = []
        for x, y, pH, pW, _ in tasks:
            ry, rx = rc(y) - RY0, rc(x) - RX0
            rh, rw = rc(y + pH) - rc(y), rc(x + pW) - rc(x)
            rh = max(1, min(rh, RH - ry))
            rw = max(1, min(rw, RW - rx))
            placements.append((ry, rx, rh, rw))

        # Gaussian weight masks (keyed by accumulator block size). The mask is
        # defined on a normalized grid, so building it at the accumulator's
        # resolution is the same falloff as at level 0.
        weight_mask_cache = {}

        def weight_for(rh, rw):
            if not has_overlap:
                return None
            key = (rh, rw)
            if key not in weight_mask_cache:
                weight_mask_cache[key] = generate_gaussian_weight_mask(rh, rw)
            return weight_mask_cache[key]

        # The overlap-count map is identical for every class, so it is built
        # once here instead of being accumulated per class inside the loop.
        count_map = np.zeros((RH, RW), dtype=np.float32)
        for ry, rx, rh, rw in placements:
            wm = weight_for(rh, rw)
            if wm is None:
                count_map[ry:ry + rh, rx:rx + rw] += 1.0
            else:
                count_map[ry:ry + rh, rx:rx + rw] += wm

        prob_map_dict = {c: np.zeros((RH, RW), dtype=np.float32) for c in class_indices}

        # ── batched inference ───────────────────────────────────────────
        img_path_for_loader = img_path if is_wsi else None
        dataset = WSIROIDataset(img_path_for_loader, full_image_array, tasks, S,
                                read_level=read_level, read_downsample=read_ds)

        # Few windows → worker startup cost outweighs the benefit; fall back to 0.
        effective_workers = args.num_workers if len(tasks) >= args.batch_size * 2 else 0
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=effective_workers,
            pin_memory=use_pin and effective_workers > 0,
            prefetch_factor=4 if effective_workers > 0 else None,
            persistent_workers=False,
            worker_init_fn=worker_init if effective_workers > 0 else None,
        )

        offset = 0
        with torch.no_grad():
            for patches_u8, bboxes, xs, ys, pHs, pWs in dataloader:
                B = patches_u8.shape[0]

                # uint8 (B, S, S, 3) -> normalized fp32 (B, 3, S, S), on device.
                pixel_values_m2f = patches_u8.to(device, non_blocking=use_pin)
                pixel_values_m2f = (pixel_values_m2f.permute(0, 3, 1, 2).float()
                                    .mul_(rescale).sub_(norm_mean).div_(norm_std))

                box = bboxes.to(device, non_blocking=use_pin) if use_box else None

                with autocast_ctx:
                    # The Swin trunk + pixel decoder see the window once; only
                    # the prompt-conditioned decoder repeats per class.
                    image_features = core.encode_image(pixel_values_m2f)
                    box_tokens = core.encode_box(box, B)

                    for c in class_indices:
                        text_tokens, text_pad_mask = prompt_cache[c]
                        seg = core.decode(
                            image_features,
                            text_tokens.expand(B, -1, -1),
                            text_pad_mask.expand(B, -1),
                            box_tokens,
                        )
                        fps = seg.float().softmax(dim=1)[:, 1].cpu().numpy()  # (B, S, S)

                        prob_map = prob_map_dict[c]
                        for k in range(B):
                            ry, rx, rh, rw = placements[offset + k]
                            fp = fps[k]
                            if fp.shape != (rh, rw):
                                fp = cv2.resize(fp, (rw, rh), interpolation=cv2.INTER_LINEAR)
                            wm = weight_for(rh, rw)
                            if wm is None:
                                prob_map[ry:ry + rh, rx:rx + rw] += fp
                            else:
                                prob_map[ry:ry + rh, rx:rx + rw] += fp * wm

                del pixel_values_m2f, image_features, box_tokens, box, patches_u8
                offset += B

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del tasks, dataset, dataloader, weight_mask_cache, placements

        # ── Otsu per class + area-sorted merge into the label map ───────
        # Each class's prob map is averaged, zeroed outside tissue (matching the
        # original prob_avg[fg_mask == 0] = 0 step), thresholded with Otsu, then
        # merged smallest-area-last so small masks overwrite large ones.
        valid = count_map > 0
        fg_roi = get_fg_roi(fg_small, sx, sy, y0, y1, x0, x1, RH, RW)
        outside_tissue = fg_roi == 0
        del fg_roi

        mask_components = []
        prob_npz = {}
        for c in list(prob_map_dict.keys()):
            prob = prob_map_dict[c]
            np.divide(prob, count_map, out=prob, where=valid)

            # Zero out probabilities outside tissue before Otsu.
            prob[outside_tissue] = 0

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

        del valid, outside_tissue, count_map

        # Sort by area descending (so smaller masks overwrite larger ones)
        mask_components.sort(key=lambda x: -x[2])

        # The label map lives on the accumulator grid; it is only expanded to
        # level 0 if --save_mask actually asks for a level-0 PNG.
        merged_mask = np.zeros((RH_full, RW_full), dtype=np.uint8)
        for c, binary_mask, _ in mask_components:
            region = merged_mask[RY0:RY0 + RH, RX0:RX0 + RW]  # view into merged_mask
            region[binary_mask] = c

        # The .jpg overview is always written; the label mask and the
        # probability maps are opt-in.
        save_thumbnail_vis(img_path, full_image_array, is_wsi, merged_mask, H, W, vis_path)

        if args.save_mask:
            save_label_png(out_path,
                           nearest_upsample(merged_mask, H, W)
                           if merged_mask.shape != (H, W) else merged_mask,
                           args.png_compression)

        if args.save_prob:
            prob_npz["prob_scale"] = np.array(prob_scale, dtype=np.int64)
            npz_path = os.path.join(args.infer_vis_dir, f'{slide_id}.npz')
            np.savez_compressed(npz_path, **prob_npz)
            print(f"Saved probability maps to {npz_path}")
        del prob_npz

        print(f"Done: {image_name} -> {vis_path}")

        del merged_mask, mask_components, prob_map_dict, fg_small
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
                        help="Number of windows per GPU forward pass. Every class is decoded "
                             "from the same encoded batch.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for parallel patch loading")
    parser.add_argument("--cpu_threads", type=int, default=8,
                        help="Cap on OpenCV/torch intra-op threads in the parent process. "
                             "Workers always run single-threaded; without a cap OpenCV "
                             "sizes its pool to the whole machine inside every worker.")
    parser.add_argument("--precision", type=str, default="tf32",
                        choices=["tf32", "fp32", "bf16", "fp16"],
                        help="Forward-pass precision. tf32 (default) keeps fp32 tensors but "
                             "runs matmuls on the tensor cores; fp32 disables that.")
    parser.add_argument("--read_level", type=str, default="auto", choices=["auto", "0"],
                        help="Pyramid level to read windows from. 'auto' reads the coarsest "
                             "level that still has every pixel the model sees (i.e. whose "
                             "downsample is <= crop_size/m2f_image_size), which decodes far "
                             "less JPEG; '0' always reads full resolution.")
    parser.add_argument("--prob_scale", type=int, default=0,
                        help="Downsample factor of the probability accumulator relative to "
                             "level 0. 0 (default) uses crop_size/m2f_image_size, i.e. the "
                             "model's own output grid; 1 accumulates at level 0.")
    parser.add_argument("--png_compression", type=int, default=1,
                        help="zlib level (0-9) for the --save_mask PNG. A level-0 "
                             "whole-slide label map is hundreds of megapixels; the "
                             "default 1 writes it in seconds instead of minutes for a "
                             "few percent more disk.")
    parser.add_argument("--save_mask", action="store_true",
                        help="Also save the label mask as <slide>.png, with pixel values "
                             "equal to the --class_names indices. Off by default; only "
                             "the <slide>.jpg overview is written.")
    parser.add_argument("--save_prob", action="store_true",
                        help="If set, also save the per-class probability maps as "
                             "<slide>.npz. Each class stores its map over the region it "
                             "covers plus a '<class>_bbox' array of [y0, y1, x0, x1] "
                             "level-0 coordinates, and 'prob_scale' gives the map's "
                             "downsample relative to level 0, so huge slides stay "
                             "memory-safe.")

    args = parser.parse_args()
    limit_cpu_threads(args.cpu_threads)
    main(args)
