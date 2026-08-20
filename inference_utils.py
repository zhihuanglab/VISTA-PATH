"""Shared plumbing for the VISTA-PATH inference entrypoints.

Both ``inference.py`` (tissue-prompted / prompt-free whole-slide inference) and
``inference_bbx.py`` (box-prompted ROI inference) build the same model and use
the same sliding-window machinery; only the prompt source and the output
post-processing differ.
"""

import contextlib
import os
import random

import cv2
import numpy as np
import torch

from models.backbones import CustomSegmentationModel

# Text encoder (PLIP = CLIP fine-tuned on pathology) and the Mask2Former
# checkpoint the segmentation trunk was initialized from. Both must match the
# values used at training time or the released weights will not load.
BASE_MODEL_NAME = "vinid/plip"
MASK2FORMER_NAME = "facebook/mask2former-swin-small-ade-semantic"
SAM_PRETRAINED = "facebook/sam-vit-base"

# Prompt template used for every class name at training time.
TEXT_TEMPLATE = "an image of {}"


def set_seed(seed=42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs (if using multi-GPU)

    torch.backends.cudnn.deterministic = True   # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


def tune_for_inference(precision="tf32", cpu_threads=8):
    """Flip the global knobs that only make sense for a forward-only run.

    ``set_seed`` pins cuDNN into deterministic / no-autotune mode, which is
    the right default for training but costs ~35% of the forward throughput.
    Inference re-runs one fixed input shape thousands of times, so autotuning
    pays for itself on the first batch.

    ``cpu_threads`` caps OpenCV and the intra-op torch pool. On a fat node
    (256 cores here) OpenCV otherwise spawns a thread per core inside every
    DataLoader worker, and the resulting oversubscription burns far more CPU
    than the resize it is parallelising.

    Returns the autocast context manager to wrap the forward pass in.
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # TF32 keeps fp32 storage but runs the matmuls/convs on the tensor cores.
    allow_tf32 = precision in ("tf32", "bf16", "fp16")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    limit_cpu_threads(cpu_threads)

    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def limit_cpu_threads(n=1):
    """Cap the thread pools that would otherwise size themselves to the whole
    machine. Called in the parent and again in every DataLoader worker."""
    n = max(1, int(n))
    cv2.setNumThreads(n)
    torch.set_num_threads(n)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


def worker_init(_worker_id):
    """DataLoader worker initializer: one OpenCV thread per worker."""
    limit_cpu_threads(1)


class SegWrapper(torch.nn.Module):
    """Thin wrapper kept for checkpoint compatibility.

    The released weights were saved from the HuggingFace Trainer, so every key
    is prefixed with ``model.``. Keeping the wrapper lets the checkpoint load
    strictly, without key surgery.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._keys_to_ignore_on_save = None

    def forward(self, pixel_values_m2f, input_ids, attention_mask, labels=None, box=None):
        logits, _ = self.model(pixel_values_m2f, input_ids, attention_mask, box)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits.detach(), "loss": loss}


def add_model_args(parser):
    """Architecture / checkpoint arguments shared by both entrypoints."""
    parser.add_argument("--checkpoint_file", type=str, required=True,
                        help="Path to the released VISTA-PATH weights "
                             "(./checkpoints/pytorch_model.bin).")
    parser.add_argument("--num_queries", type=int, default=20,
                        help="Mask2Former queries. Must match training (20 for the "
                             "released checkpoint).")
    parser.add_argument("--m2f_image_size", type=int, default=512,
                        help="Resolution fed to the model. Must match the value used "
                             "at training time (512 for the released checkpoint).")
    parser.add_argument("--tune_mode", type=str, default="freeze")
    # Kept for backwards compatibility with the published command lines; the
    # current architecture derives its widths from the pretrained trunks.
    parser.add_argument("--d_model", type=int, default=512, help="Unused, kept for CLI compatibility.")
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4, help="Unused, kept for CLI compatibility.")
    return parser


def load_model(args, device, bbx_random=None):
    """Instantiate VISTA-PATH and load the released checkpoint.

    ``bbx_random`` overrides ``args.bbx_random``; it is the probability of
    *dropping* the box prompt. Use 0 to always honour the box and 1 to run
    fully prompt-free.
    """
    core_model = CustomSegmentationModel(
        BASE_MODEL_NAME,
        args.d_model,
        args.nhead,
        args.num_layers,
        args.bbx_random if bbx_random is None else bbx_random,
        args.tune_mode,
        mask2former_name=MASK2FORMER_NAME,
        num_queries=args.num_queries,
        image_size=args.m2f_image_size,
        sam_pretrained=SAM_PRETRAINED,
    )
    model = SegWrapper(core_model)

    checkpoint_path = args.checkpoint_file
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Accept both a bare state_dict and a training checkpoint dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    # Weights saved from a plain CustomSegmentationModel (no Trainer wrapper)
    # have no "model." prefix; add it so both layouts load strictly.
    if not any(k.startswith("model.") for k in checkpoint):
        checkpoint = {f"model.{k}": v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def encode_class_texts(processor, idx_to_class, class_indices):
    """Tokenize the prompt for every class once; workers reuse the cache."""
    text_cache = {}
    for c in class_indices:
        text = TEXT_TEMPLATE.format(idx_to_class[c])
        enc = processor(text=text, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=77)
        text_cache[c] = (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0))
    return text_cache


@torch.no_grad()
def encode_class_prompts(core_model, processor, idx_to_class, class_indices, device):
    """Run the PLIP text tower once per class name and keep the projected
    token sequence on the GPU.

    The text branch depends only on the class name, so a whole-slide run that
    re-tokenizes and re-encodes the same handful of prompts for every batch is
    doing the same work thousands of times. Each entry is a
    ``(1, 77, C)`` token tensor plus its pad mask, ready to be expanded to the
    batch size.
    """
    token_cache = encode_class_texts(processor, idx_to_class, class_indices)
    prompt_cache = {}
    for c in class_indices:
        input_ids, attention_mask = token_cache[c]
        tokens, pad_mask = core_model.encode_text(
            input_ids[None].to(device), attention_mask[None].to(device),
        )
        prompt_cache[c] = (tokens, pad_mask)
    return prompt_cache


# OpenCV keeps 16-bit signed coordinates internally in several resize paths,
# so anything past SHRT_MAX in either axis takes the numpy route.
_CV_DIM_LIMIT = 32767


def nearest_upsample(src, out_h, out_w):
    """Nearest-neighbour resample of ``src`` onto an ``out_h x out_w`` grid.

    ``cv2.resize`` is ~3x faster than fancy indexing and avoids materializing
    the intermediate row gather, but it caps out on very large destinations —
    a level-0 whole-slide mask can exceed OpenCV's coordinate range — so the
    index form is kept as the fallback.
    """
    h, w = src.shape[:2]
    if (h, w) == (out_h, out_w):
        return src
    if max(out_h, out_w, h, w) < _CV_DIM_LIMIT:
        return cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    yi = np.minimum((np.arange(out_h) * (h / out_h)).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(out_w) * (w / out_w)).astype(np.int64), w - 1)
    return src[yi][:, xi]


def get_patch_positions(H, W, patch_size=1024, overlap=128):
    """Compute sliding-window patch top-left positions without loading image data."""
    overlap = max(0, min(overlap, patch_size - 1))
    stride_y = patch_size - overlap if H >= patch_size else H
    stride_x = patch_size - overlap if W >= patch_size else W

    def build_coords(limit, psize, stride):
        if limit <= psize:
            return [0]
        coords = list(range(0, limit - psize + 1, stride))
        if coords[-1] + psize < limit:
            coords.append(limit - psize)
        return coords

    y_coords = build_coords(H, min(H, patch_size), stride_y)
    x_coords = build_coords(W, min(W, patch_size), stride_x)
    return [(x, y) for y in y_coords for x in x_coords]


def generate_gaussian_weight_mask(height, width, sigma_scale=0.125):
    """Gaussian falloff toward patch edges, used to blend overlapping windows."""
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xv, yv = np.meshgrid(x, y)
    sigma_x = sigma_scale * 2
    sigma_y = sigma_scale * 2
    gauss = np.exp(-((xv**2) / (2 * sigma_x**2) + (yv**2) / (2 * sigma_y**2)))
    gauss /= gauss.max()
    return gauss.astype(np.float32)
