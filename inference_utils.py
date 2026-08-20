"""Shared plumbing for the VISTA-PATH inference entrypoints.

Both ``inference.py`` (tissue-prompted / prompt-free whole-slide inference) and
``inference_bbx.py`` (box-prompted ROI inference) build the same model and use
the same sliding-window machinery; only the prompt source and the output
post-processing differ.
"""

import random

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
