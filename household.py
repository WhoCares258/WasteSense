#!/usr/bin/env python3
"""
EVA02-Base 448 on Household Garbage / Garbage Classification (12 classes)

Protocol:
- Five independent stratified 80/10/10 train/val/test splits
- Split seeds: 0, 1, 2, 3, 4
- Fixed training seed: 42
- Same EVA02-Base training recipe used for TrashNet / RealWaste
- All five seeds run back-to-back on one GPU

HPC optimizations:
- Decode every source image ONCE into RAM at startup
- Preserve the original timm training transform exactly
- Store large active checkpoints only in /tmp
- Persist only small JSON/CSV artifacts in home storage
- Skip completed seeds automatically on rerun

Default dataset root:
    ~/garbage_classification

Recommended for your current 4-core / 1x L4 node:
    workers=3
    eval_workers=1
    prefetch=4

Run:
    source venv/bin/activate
    python train_eva02_base_household_80_10_10_hpc.py
"""

# =====================================================================
# LOCAL CACHES BEFORE TIMM IMPORT
# =====================================================================

import os

LOCAL_SCRATCH = os.environ.get("TMPDIR", "/tmp")

os.environ.setdefault("HF_HOME", os.path.join(LOCAL_SCRATCH, "hf_home"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(LOCAL_SCRATCH, "hf_home", "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(LOCAL_SCRATCH, "torch_home"))

import argparse
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import create_transform

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from safetensors.torch import load_file as load_safetensors


# =====================================================================
# EVA02-BASE METHOD
# =====================================================================

MODEL_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
HF_REPO = "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"

SPLIT_SEEDS = [1, 2, 3, 4]
TRAINING_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 30

PHASE1_LR = 1e-3
PHASE2_BASE_LR = 8e-6
PHASE2_HEAD_LR = 4e-5
PHASE2_WARMUP_EPOCHS = 2

LAYER_DECAY = 0.75
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

AMP_DTYPE = torch.float16
USE_CHANNELS_LAST = True
USE_GRAD_CHECKPOINTING = True

DATA_CONFIG = {
    "input_size": (3, 448, 448),
    "mean": (0.48145466, 0.4578275, 0.40821073),
    "std": (0.26862954, 0.26130258, 0.27577711),
    "interpolation": "bicubic",
    "crop_pct": 1.0,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = TRAINING_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =====================================================================
# DATASET DISCOVERY / RAM CACHE
# =====================================================================

def discover_dataset(root):
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    if not class_names:
        raise RuntimeError(f"No class folders found under {root}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    rows = []

    for class_name in class_names:
        class_dir = root / class_name

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                rows.append(
                    {
                        "path": str(path),
                        "label": class_to_idx[class_name],
                        "class": class_name,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No images found under {root}")

    return df, class_names


def predecode_to_ram(df):
    """
    Decode every image once into RGB uint8 RAM.
    No pre-resize is done here, so the original timm resize/crop/augmentation
    pipeline remains unchanged.
    """

    print("\n" + "=" * 90)
    print("PREDECODING DATASET INTO RAM")
    print("=" * 90)

    decoded_images = []
    total_bytes = 0
    t0 = time.time()

    for i, path_str in enumerate(df["path"].tolist(), start=1):
        with Image.open(path_str) as img:
            arr = np.asarray(
                img.convert("RGB"),
                dtype=np.uint8,
            ).copy()

        decoded_images.append(arr)
        total_bytes += arr.nbytes

        if i % 1000 == 0 or i == len(df):
            print(
                f"Decoded {i:6d}/{len(df)} | "
                f"RAM {total_bytes / 1024**3:.2f} GiB | "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    print(
        f"Predecode complete: "
        f"{len(decoded_images)} images | "
        f"{total_bytes / 1024**3:.2f} GiB raw RGB | "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    return decoded_images


# =====================================================================
# STRATIFIED 80 / 10 / 10
# =====================================================================

def make_split(df, split_seed):
    indices = np.arange(len(df))
    labels = df["label"].to_numpy()

    # 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=split_seed,
        shuffle=True,
        stratify=labels,
    )

    # Temp is split evenly into 10% val and 10% test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=split_seed,
        shuffle=True,
        stratify=labels[temp_idx],
    )

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
        np.asarray(test_idx),
    )


# =====================================================================
# ORIGINAL TIMM TRANSFORMS
# =====================================================================

def build_transforms():
    train_transform = create_transform(
        input_size=DATA_CONFIG["input_size"],
        is_training=True,
        mean=DATA_CONFIG["mean"],
        std=DATA_CONFIG["std"],
        interpolation=DATA_CONFIG["interpolation"],
        hflip=0.5,
        color_jitter=0.08,
        re_prob=0.10,
        re_mode="pixel",
        re_count=1,
    )

    eval_transform = create_transform(
        input_size=DATA_CONFIG["input_size"],
        is_training=False,
        mean=DATA_CONFIG["mean"],
        std=DATA_CONFIG["std"],
        interpolation=DATA_CONFIG["interpolation"],
        crop_pct=DATA_CONFIG["crop_pct"],
    )

    return train_transform, eval_transform


class CachedDataset(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_idx = int(self.indices[item])

        image = Image.fromarray(
            self.images[source_idx],
            mode="RGB",
        )

        image = self.transform(image)

        return image, int(self.labels[source_idx])


def make_loader(
    dataset,
    train,
    workers,
    prefetch,
    persistent=True,
):
    generator = torch.Generator()
    generator.manual_seed(TRAINING_SEED)

    kwargs = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker if workers > 0 else None,
        persistent_workers=(persistent if workers > 0 else False),
        generator=(generator if train else None),
    )

    if workers > 0:
        kwargs["prefetch_factor"] = prefetch

    return DataLoader(**kwargs)


# =====================================================================
# MODEL / PRETRAINED WEIGHTS
# =====================================================================

def get_pretrained_path():
    scratch_dir = Path(LOCAL_SCRATCH) / f"eva02_base_weights_{os.getuid()}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    target = scratch_dir / "model.safetensors"

    if target.exists():
        return target

    candidates = [
        Path.home() / "pretrained/eva02_base_448/model.safetensors",
        Path("/home/user/22104053/pretrained/eva02_base_448/model.safetensors"),
    ]

    for path in candidates:
        if path.exists():
            print(f"[MODEL] Copying weights to /tmp from {path}", flush=True)
            shutil.copy2(path, target)
            return target

    print("[MODEL] Downloading weights to /tmp", flush=True)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename="model.safetensors",
        local_dir=str(scratch_dir),
    )

    return Path(downloaded)


def clean_key_candidates(key):
    candidates = [key]

    for prefix in ["module.", "_orig_mod.", "model."]:
        if key.startswith(prefix):
            candidates.append(key[len(prefix):])

    return candidates


def create_model(num_classes, pretrained_path):
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )

    if USE_GRAD_CHECKPOINTING and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)

    pretrained_state = load_safetensors(str(pretrained_path))
    model_state = model.state_dict()

    filtered_state = {}

    loaded = 0
    skipped = 0

    for key, value in pretrained_state.items():
        matched = False

        for candidate_key in clean_key_candidates(key):
            if (
                candidate_key in model_state
                and value.shape == model_state[candidate_key].shape
            ):
                filtered_state[candidate_key] = value
                loaded += 1
                matched = True
                break

        if not matched:
            skipped += 1

    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state,
        strict=False,
    )

    print(
        f"[MODEL] loaded={loaded} "
        f"skipped={skipped} "
        f"missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)}",
        flush=True,
    )

    return model


# =====================================================================
# OPTIMIZER HELPERS
# =====================================================================

def freeze_backbone_train_head(model):
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name or "fc" in name:
                param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def get_layer_id(name, num_blocks):
    if name.startswith("head"):
        return num_blocks + 2

    if name.startswith("norm"):
        return num_blocks + 1

    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1]) + 1

    return 0


def build_simple_param_groups(model, lr):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            "params": decay,
            "lr": lr,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": no_decay,
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]


def build_layer_decay_param_groups(model):
    num_blocks = len(model.blocks) if hasattr(model, "blocks") else 12
    max_layer_id = num_blocks + 2

    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
        )

        wd = 0.0 if no_decay else WEIGHT_DECAY

        if name.startswith("head"):
            lr = PHASE2_HEAD_LR
        else:
            layer_id = get_layer_id(name, num_blocks)
            lr_scale = LAYER_DECAY ** (max_layer_id - layer_id)
            lr = PHASE2_BASE_LR * lr_scale

        key = (lr, wd)

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
            }

        groups[key]["params"].append(param)

    return list(groups.values())


def create_scheduler(
    optimizer,
    epochs,
    warmup_epochs=0,
):
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.10,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# =====================================================================
# TRAIN / EVAL
# =====================================================================

def move_batch(images, labels, device):
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if USE_CHANNELS_LAST:
        images = images.contiguous(
            memory_format=torch.channels_last
        )

    return images, labels


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = move_batch(images, labels, device)

        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=AMP_DTYPE,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(
            loss / ACCUMULATION_STEPS
        ).backward()

        should_step = (
            batch_idx % ACCUMULATION_STEPS == 0
            or batch_idx == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += (
            loss.item()
            * labels.size(0)
        )

    return (
        total_loss / total,
        100.0 * correct / total,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in loader:
        images, labels = move_batch(
            images,
            labels,
            device,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=AMP_DTYPE,
        ):
            outputs = model(images)
            loss = criterion(
                outputs,
                labels,
            )

        preds = outputs.argmax(dim=1)

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_preds.extend(
            preds.cpu().tolist()
        )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    metrics = {
        "accuracy":
            100.0 * accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_precision":
            100.0 * precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            100.0 * recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "macro_f1":
            100.0 * f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "weighted_f1":
            100.0 * f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
    }

    return (
        total_loss / total,
        metrics,
        y_true,
        y_pred,
    )


# =====================================================================
# ONE SEED
# =====================================================================

def run_seed(
    seed,
    df,
    images,
    classes,
    train_transform,
    eval_transform,
    pretrained_path,
    output_root,
    device,
    workers,
    eval_workers,
    prefetch,
):
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_metrics_path = (
        seed_dir
        / "final_metrics.json"
    )

    # Completed-seed resume.
    if final_metrics_path.exists():
        print(
            f"[SKIP] seed {seed} already complete",
            flush=True,
        )

        with open(final_metrics_path) as f:
            return json.load(f)

    train_idx, val_idx, test_idx = make_split(
        df,
        seed,
    )

    labels = df["label"].to_numpy(
        dtype=np.int64
    )

    print(
        f"\nSEED {seed} | "
        f"train {len(train_idx)} "
        f"val {len(val_idx)} "
        f"test {len(test_idx)}",
        flush=True,
    )

    with open(
        seed_dir / "split_indices.json",
        "w",
    ) as f:
        json.dump(
            {
                "split_seed": seed,
                "training_seed": TRAINING_SEED,
                "split_protocol": "stratified_80_10_10",
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "class_names": classes,
            },
            f,
        )

    train_loader = make_loader(
        CachedDataset(
            images,
            labels,
            train_idx,
            train_transform,
        ),
        True,
        workers,
        prefetch,
        True,
    )

    val_loader = make_loader(
        CachedDataset(
            images,
            labels,
            val_idx,
            eval_transform,
        ),
        False,
        eval_workers,
        max(2, min(prefetch, 4)),
        True,
    )

    test_loader = make_loader(
        CachedDataset(
            images,
            labels,
            test_idx,
            eval_transform,
        ),
        False,
        eval_workers,
        max(2, min(prefetch, 4)),
        False,
    )

    set_seed(TRAINING_SEED)

    model = create_model(
        len(classes),
        pretrained_path,
    ).to(device)

    if USE_CHANNELS_LAST:
        model = model.to(
            memory_format=torch.channels_last
        )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    temp_seed_dir = (
        Path(LOCAL_SCRATCH)
        / f"eva02_household_801010_seed_{seed}_{os.getuid()}"
    )

    temp_seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        temp_seed_dir
        / "best_model.pt"
    )

    phases = [
        (
            "phase1_head_only",
            PHASE1_EPOCHS,
            True,
            0,
        ),
        (
            "phase2_full_finetune",
            PHASE2_EPOCHS,
            False,
            PHASE2_WARMUP_EPOCHS,
        ),
    ]

    best_val_accuracy = 0.0
    history = []
    seed_start = time.time()

    for phase_index, (
        phase_name,
        phase_epochs,
        freeze,
        warmup_epochs,
    ) in enumerate(phases):

        if freeze:
            freeze_backbone_train_head(
                model
            )

            param_groups = (
                build_simple_param_groups(
                    model,
                    PHASE1_LR,
                )
            )

        else:
            unfreeze_all(model)

            param_groups = (
                build_layer_decay_param_groups(
                    model
                )
            )

        optimizer = optim.AdamW(
            param_groups
        )

        scheduler = create_scheduler(
            optimizer,
            phase_epochs,
            warmup_epochs,
        )

        for epoch in range(
            1,
            phase_epochs + 1,
        ):
            epoch_start = time.time()

            (
                train_loss,
                train_accuracy,
            ) = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
            )

            (
                val_loss,
                val_metrics,
                _,
                _,
            ) = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Same original behavior.
            scheduler.step()

            elapsed = (
                time.time()
                - epoch_start
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"[Seed {seed}] "
                f"{phase_name} "
                f"{epoch:02d}/{phase_epochs} | "
                f"train {train_loss:.4f} "
                f"{train_accuracy:.2f}% | "
                f"val {val_loss:.4f} "
                f"{val_metrics['accuracy']:.2f}% | "
                f"F1 {val_metrics['macro_f1']:.2f}% | "
                f"{elapsed:.1f}s",
                flush=True,
            )

            history.append(
                {
                    "phase": phase_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_precision": val_metrics["macro_precision"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "lr": current_lr,
                    "epoch_seconds": elapsed,
                }
            )

            # Original checkpoint rule:
            # highest validation accuracy, later ties replace earlier.
            if (
                val_metrics["accuracy"]
                >= best_val_accuracy
            ):
                best_val_accuracy = (
                    val_metrics["accuracy"]
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "epoch":
                            epoch,

                        "phase":
                            phase_name,

                        "best_val_accuracy":
                            best_val_accuracy,

                        "best_val_loss":
                            val_loss,
                    },
                    best_model_path,
                )

                print(
                    f"  -> BEST "
                    f"{best_val_accuracy:.2f}%",
                    flush=True,
                )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    (
        test_loss,
        test_metrics,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    result = {
        "split_seed": int(seed),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_precision": float(test_metrics["macro_precision"]),
        "macro_recall": float(test_metrics["macro_recall"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(checkpoint["best_val_accuracy"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "best_epoch": int(checkpoint["epoch"]),
        "best_phase": checkpoint["phase"],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "seed_seconds": float(time.time() - seed_start),
        "split_protocol": "stratified_80_10_10",
    }

    with open(
        final_metrics_path,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    pd.DataFrame(
        history
    ).to_csv(
        seed_dir / "training_log.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        seed_dir
        / "confusion_matrix.csv"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        seed_dir
        / "classification_report.csv"
    )

    print(
        f"SEED {seed} TEST "
        f"accuracy={result['accuracy']:.4f}% "
        f"macro_f1={result['macro_f1']:.4f}% "
        f"time={result['seed_seconds']/60:.1f} min",
        flush=True,
    )

    # Remove active large checkpoint immediately.
    shutil.rmtree(
        temp_seed_dir,
        ignore_errors=True,
    )

    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del val_loader
    del test_loader

    gc.collect()
    torch.cuda.empty_cache()

    return result


# =====================================================================
# SUMMARY
# =====================================================================

def save_summary(
    output_root,
    seeds,
):
    rows = []

    for seed in seeds:
        metrics_path = (
            output_root
            / f"seed_{seed}"
            / "final_metrics.json"
        )

        if metrics_path.exists():
            with open(metrics_path) as f:
                rows.append(
                    json.load(f)
                )

    if not rows:
        return

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "split_seed"
        )
    )

    results.to_csv(
        output_root
        / "all_split_results.csv",
        index=False,
    )

    if len(results) != len(seeds):
        return

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    summary = []

    print("\nMEAN ± SD", flush=True)

    for metric in metrics:
        values = results[
            metric
        ].to_numpy(
            dtype=float
        )

        mean = values.mean()

        sd = values.std(
            ddof=1
        )

        half_width = (
            2.7764451051977987
            * sd
            / math.sqrt(
                len(values)
            )
        )

        print(
            f"{metric:20s}: "
            f"{mean:.4f} ± "
            f"{sd:.4f}% "
            f"[95% CI "
            f"{mean-half_width:.4f}, "
            f"{mean+half_width:.4f}]",
            flush=True,
        )

        summary.append(
            {
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )

    pd.DataFrame(
        summary
    ).to_csv(
        output_root
        / "summary_mean_sd_ci.csv",
        index=False,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(
            Path.home()
            / "garbage_classification"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            Path.home()
            / "household_garbage_eva02_80_10_10_results"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Training workers. Default 3 for 4-core node.",
    )

    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SPLIT_SEEDS,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected."
        )

    data_root = (
        Path(args.data)
        .expanduser()
        .resolve()
    )

    output_root = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda:0"
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass

    set_seed(
        TRAINING_SEED
    )

    print(
        "=" * 100
    )
    print(
        "EVA02-BASE 448 | HOUSEHOLD GARBAGE | STRATIFIED 80/10/10"
    )
    print(
        "=" * 100
    )

    print(
        f"Data root          : {data_root}"
    )
    print(
        f"Output             : {output_root}"
    )
    print(
        f"GPU                : {torch.cuda.get_device_name(0)}"
    )
    print(
        f"CPU count          : {os.cpu_count()}"
    )
    print(
        f"Workers            : {args.workers}"
    )
    print(
        f"Eval workers       : {args.eval_workers}"
    )
    print(
        f"Prefetch           : {args.prefetch}"
    )
    print(
        f"Batch x accum      : "
        f"{BATCH_SIZE} x {ACCUMULATION_STEPS} "
        f"= {EFFECTIVE_BATCH_SIZE}"
    )
    print(
        "Split protocol     : stratified 80/10/10"
    )
    print(
        f"Split seeds        : {args.seeds}"
    )
    print(
        f"Training seed      : {TRAINING_SEED}"
    )
    print(
        "Large checkpoints  : /tmp only"
    )

    df, classes = discover_dataset(
        data_root
    )

    print(
        "\nClass counts:"
    )

    print(
        df["class"]
        .value_counts()
        .reindex(classes)
        .to_string()
    )

    print(
        f"\nTotal images: {len(df)}"
    )

    images = predecode_to_ram(
        df
    )

    (
        train_transform,
        eval_transform,
    ) = build_transforms()

    pretrained_path = (
        get_pretrained_path()
    )

    for seed in args.seeds:
        run_seed(
            seed,
            df,
            images,
            classes,
            train_transform,
            eval_transform,
            pretrained_path,
            output_root,
            device,
            args.workers,
            args.eval_workers,
            args.prefetch,
        )

        save_summary(
            output_root,
            args.seeds,
        )

    save_summary(
        output_root,
        args.seeds,
    )

    print(
        f"\nDone. Results: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
