"""Pre-train VISTA-PATH with the HuggingFace Trainer.

Launch with accelerate, e.g. on 4 GPUs:

    accelerate launch --num_processes 4 train.py \
        --csv_file train.csv --json_dir ./idx_to_names --output_dir ./checkpoints/run

See Scripts/train.sh for the configuration used for the released checkpoint.
"""
import csv
import gc
import logging
import os
from argparse import ArgumentParser
from math import ceil

import numpy as np
import torch
import torch.distributed as dist
from transformers import (AutoImageProcessor, CLIPProcessor, Trainer,
                          TrainerCallback, TrainingArguments)

from dataset.datasets import PathSegDataset
from models.backbones import CustomSegmentationModel

# PyTorch 2.6+ loads with weights_only=True by default, which rejects the numpy
# objects in the RNG state saved with each checkpoint. Allow them so --resume works.
try:
    import numpy._core.multiarray
    import numpy.dtypes
    torch.serialization.add_safe_globals([
        numpy._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        *[v for v in vars(numpy.dtypes).values() if isinstance(v, type)],
    ])
except (ImportError, AttributeError):
    pass

BASE_MODEL = "vinid/plip"
MASK2FORMER_NAME = "facebook/mask2former-swin-small-ade-semantic"


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_logging(output_dir):
    if is_main_process():
        logging.basicConfig(
            filename=os.path.join(output_dir, "training_info.log"),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        logging.getLogger().addHandler(logging.StreamHandler())


class LoggingCallback(TrainerCallback):
    """Writes the training loss and learning rate to the log and to a CSV."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        if is_main_process():
            with open(self.csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(["epoch", "step", "loss", "learning_rate"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        loss, lr = logs.get("loss"), logs.get("learning_rate")
        msg = f"[Epoch {state.epoch:.2f} | Step {state.global_step}]"
        if loss is not None:
            msg += f" Loss: {loss:.4f}"
        if lr is not None:
            msg += f", LR: {lr:.2e}"
        logging.info(msg)
        with open(self.csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                state.epoch, state.global_step,
                "" if loss is None else f"{loss:.4f}",
                "" if lr is None else f"{lr:.6e}",
            ])


class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()


def collate_segmentation(features):
    pixel_values_m2f, input_ids, attention_mask, masks, _text, boxes = zip(*features)
    return {
        "pixel_values_m2f": torch.stack(pixel_values_m2f),
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(masks),
        "box": torch.stack(boxes),
    }


class SegWrapper(torch.nn.Module):
    """Adapts CustomSegmentationModel to the Trainer's dict-in / loss-out interface."""

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


def main(args):
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)
    # Crops are already image_size x image_size, so the Mask2Former processor
    # only rescales and ImageNet-normalizes.
    m2f_processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_NAME, do_resize=False, do_reduce_labels=False, ignore_index=255,
    )

    train_dataset = PathSegDataset(
        args.csv_file, args.json_dir, processor, "train",
        (args.crop_size, args.crop_size),
        augmentations=args.augmentations,
        image_size=args.m2f_image_size,
        m2f_processor=m2f_processor,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = args.batch_size * world_size * args.grad_accum_steps
    steps_per_epoch = ceil(len(train_dataset) / effective_batch)
    logging_steps = max(10, steps_per_epoch // 5)  # ~5 logs per epoch

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=logging_steps,
        logging_strategy="steps",
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_safetensors=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    if is_main_process():
        print(f"Training samples: {len(train_dataset)}")
        print(f"Per-device batch: {args.batch_size} | processes: {world_size} | "
              f"grad accum: {args.grad_accum_steps} | effective batch: {effective_batch}")
        print(f"Steps per epoch: {steps_per_epoch} | total steps: {steps_per_epoch * args.num_epochs}")
        print(f"LR: {args.learning_rate} | weight decay: {args.weight_decay} | warmup ratio: {args.warmup_ratio}")

    core_model = CustomSegmentationModel(
        BASE_MODEL, args.d_model, args.nhead, args.num_layers, args.bbx_random,
        args.tune_mode, mask2former_name=MASK2FORMER_NAME,
        num_queries=args.num_queries, image_size=args.m2f_image_size,
    )
    model = SegWrapper(core_model)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info("Starting training")

    last_checkpoint = None
    if args.resume:
        checkpoints = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir)
                       if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            logging.info(f"Resuming from {last_checkpoint}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_segmentation,
        callbacks=[
            LoggingCallback(os.path.join(args.output_dir, "train_log.csv")),
            MemoryCleanupCallback(),
        ],
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Training CSV with train_img, train_mask, train_idx columns.")
    parser.add_argument("--json_dir", type=str, default="./idx_to_names",
                        help="Directory of <dataset>.json class-index-to-name maps.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/train")
    parser.add_argument("--tune_mode", type=str, default="freeze",
                        help="'freeze' keeps the PLIP text encoder frozen.")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Side of the random crop taken from each image, in image pixels.")
    parser.add_argument("--m2f_image_size", type=int, default=512,
                        help="Resolution the crop is resized to before Mask2Former.")
    parser.add_argument("--bbx_random", type=float, default=0.5,
                        help="Probability that a batch is trained on the text prompt alone, without boxes.")
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--augmentations", action="store_true",
                        help="Random flips and 90-degree rotations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Keep at most this many checkpoints (default: keep all).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint-* in output_dir.")
    main(parser.parse_args())
