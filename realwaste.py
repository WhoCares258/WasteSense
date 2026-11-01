#!/usr/bin/env python3
"""RealWaste EVA02-Base repeated stratified holdout training.

Aligned with the training recipe used by the WasteSense TrashNet and Household
Garbage scripts: timm transforms, EVA02 normalization, two-stage training,
epoch-based cosine schedules, and validation-accuracy checkpoint selection.

This is a harmonized reproduction implementation, not proof that the exact
historical RealWaste image transforms or stochastic execution are recoverable.
No previously reported experimental results are changed by this file.

Requirements: torch, timm, scikit-learn, numpy, pandas, pillow.
Examples:
    python realwaste.py --data-dir ~/RealWaste --seeds 0 1 2 3 4
    python realwaste.py --data-dir ~/RealWaste --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm
from timm.data import create_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
TRAINING_SEED = 42
SPLIT_SEEDS = [0, 1, 2, 3, 4]
SPLIT_COUNTS = (3326, 617, 809)
EXPECTED_CLASSES = {
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',
    'Paper', 'Plastic', 'Textile Trash', 'Vegetation',
}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 30
PHASE1_LR = 1e-3
PHASE2_BACKBONE_LR = 8e-6
PHASE2_HEAD_LR = 4e-5
PHASE2_WARMUP_EPOCHS = 2
LAYER_DECAY = 0.75
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
ACCUMULATION_STEPS = 2
AMP_DTYPE = torch.float16

DATA_CONFIG = {
    'input_size': (3, 448, 448),
    'mean': (0.48145466, 0.4578275, 0.40821073),
    'std': (0.26862954, 0.26130258, 0.27577711),
    'interpolation': 'bicubic',
    'crop_pct': 1.0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = TRAINING_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path,
                        default=Path.home() / 'realwaste_eva02_stratified_saved')
    parser.add_argument('--seeds', nargs='+', type=int,
                        choices=SPLIT_SEEDS, default=SPLIT_SEEDS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval-workers', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum', type=int, default=ACCUMULATION_STEPS)
    parser.add_argument('--head-epochs', type=int, default=PHASE1_EPOCHS)
    parser.add_argument('--finetune-epochs', type=int, default=PHASE2_EPOCHS)
    parser.add_argument('--resume', action='store_true',
                        help='Skip completed splits; does not resume a partial epoch.')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Debugging only; not part of the paper protocol.')
    return parser.parse_args()


def find_data_root(requested: Path | None) -> Path:
    if requested is not None:
        candidates = [requested.expanduser()]
    else:
        home = Path.home()
        candidates = [home / value for value in (
            'RealWaste', 'realwaste', 'RealWaste-main', 'datasets/RealWaste',
            'datasets/realwaste', 'data/RealWaste', 'realwaste_dataset',
        )]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        folders = [candidate] + [p for p in candidate.iterdir() if p.is_dir()]
        for folder in folders:
            if EXPECTED_CLASSES.issubset({p.name for p in folder.iterdir() if p.is_dir()}):
                return folder
    raise FileNotFoundError(
        'Could not locate the nine RealWaste category folders. '
        'Pass --data-dir /path/to/RealWaste.'
    )


def discover_dataset(root: Path):
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if set(classes) != EXPECTED_CLASSES:
        raise ValueError(f'Expected 9 RealWaste categories, found {classes}')
    paths = []
    labels = []
    for class_index, class_name in enumerate(classes):
        for path in sorted((root / class_name).rglob('*')):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(path)
                labels.append(class_index)
    if len(paths) != sum(SPLIT_COUNTS):
        raise ValueError(f'Expected 4752 images, found {len(paths)}')
    return paths, np.asarray(labels, dtype=np.int64), classes


def predecode_to_ram(paths):
    images = []
    t0 = time.time()
    for index, path in enumerate(paths, 1):
        with Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB'), dtype=np.uint8).copy())
        if index % 500 == 0 or index == len(paths):
            print(f'[CACHE] {index}/{len(paths)} images in {time.time()-t0:.1f}s', flush=True)
    return images


def build_transforms():
    # Matches the explicitly configured transforms in trashnet.py/household.py.
    train_transform = create_transform(
        input_size=DATA_CONFIG['input_size'], is_training=True,
        mean=DATA_CONFIG['mean'], std=DATA_CONFIG['std'],
        interpolation=DATA_CONFIG['interpolation'],
        hflip=0.5, color_jitter=0.08,
        re_prob=0.10, re_mode='pixel', re_count=1,
    )
    eval_transform = create_transform(
        input_size=DATA_CONFIG['input_size'], is_training=False,
        mean=DATA_CONFIG['mean'], std=DATA_CONFIG['std'],
        interpolation=DATA_CONFIG['interpolation'],
        crop_pct=DATA_CONFIG['crop_pct'],
    )
    return train_transform, eval_transform


class RAMDataset(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_index = int(self.indices[item])
        image = Image.fromarray(self.images[source_index])
        return self.transform(image), int(self.labels[source_index]), source_index


def make_loader(dataset, train: bool, batch_size: int, workers: int, prefetch: int):
    generator = torch.Generator()
    generator.manual_seed(TRAINING_SEED)
    kwargs = dict(
        dataset=dataset, batch_size=batch_size, shuffle=train,
        drop_last=False, num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0),
        worker_init_fn=(seed_worker if workers > 0 else None),
        generator=(generator if train else None),
    )
    if workers > 0:
        kwargs['prefetch_factor'] = prefetch
    return DataLoader(**kwargs)


def make_split(labels, split_seed: int):
    # Preserve exact RealWaste split sizes (3326/617/809).
    all_indices = np.arange(len(labels))
    train_indices, remaining = train_test_split(
        all_indices, train_size=SPLIT_COUNTS[0], random_state=split_seed,
        shuffle=True, stratify=labels,
    )
    val_indices, test_indices = train_test_split(
        remaining, train_size=SPLIT_COUNTS[1], random_state=split_seed,
        shuffle=True, stratify=labels[remaining],
    )
    if tuple(map(len, (train_indices, val_indices, test_indices))) != SPLIT_COUNTS:
        raise AssertionError('Unexpected train/validation/test sizes')
    if len(set(train_indices) | set(val_indices) | set(test_indices)) != len(labels):
        raise AssertionError('Split overlap or missing samples')
    return train_indices, val_indices, test_indices


def create_model(num_classes: int, pretrained: bool):
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=True)
    return model


def freeze_backbone_train_head(model):
    for param in model.parameters():
        param.requires_grad_(False)
    if not hasattr(model, 'head'):
        raise RuntimeError('Expected EVA02 classifier at model.head')
    for param in model.head.parameters():
        param.requires_grad_(True)


def build_simple_param_groups(model, lr: float):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = (param.ndim <= 1 or name.endswith('.bias') or
                    'pos_embed' in name or 'cls_token' in name)
        (no_decay_params if no_decay else decay_params).append(param)
    return [
        {'params': decay_params, 'lr': lr, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0},
    ]


def layer_id(name: str, nblocks: int):
    if name.startswith('head'):
        return nblocks + 2
    if name.startswith('norm'):
        return nblocks + 1
    if name.startswith('blocks.'):
        pieces = name.split('.')
        if len(pieces) > 1 and pieces[1].isdigit():
            return int(pieces[1]) + 1
    return 0


def build_layer_decay_param_groups(model):
    nblocks = len(model.blocks)
    max_layer_id = nblocks + 2
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = (param.ndim <= 1 or name.endswith('.bias') or
                    'pos_embed' in name or 'cls_token' in name)
        weight_decay = 0.0 if no_decay else WEIGHT_DECAY
        if name.startswith('head'):
            lr = PHASE2_HEAD_LR
        else:
            lr = PHASE2_BACKBONE_LR * (LAYER_DECAY **
                                        (max_layer_id - layer_id(name, nblocks)))
        key = (lr, weight_decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr': lr, 'weight_decay': weight_decay}
        groups[key]['params'].append(param)
    return list(groups.values())


def create_scheduler(optimizer, epochs: int, warmup_epochs: int = 0):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.10, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, accum: int):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    count = 0
    for batch_number, (images, labels, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            logits = model(images)
            loss = criterion(logits, labels)
        # Match TrashNet and Household gradient accumulation behavior.
        scaler.scale(loss / accum).backward()
        if batch_number % accum == 0 or batch_number == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        count += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / count, 100.0 * correct / count


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    count = 0
    targets, predictions, indices, probabilities = [], [], [], []
    for images, labels, source_ids in loader:
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        labels_gpu = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            logits = model(images)
            loss = criterion(logits, labels_gpu)
        probabilities_batch = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probabilities_batch.argmax(axis=1)
        total_loss += loss.item() * labels.size(0)
        count += labels.size(0)
        targets.extend(labels.numpy().tolist())
        predictions.extend(preds.tolist())
        indices.extend(source_ids.numpy().tolist())
        probabilities.append(probabilities_batch)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    metrics = {
        'loss': total_loss / count,
        'accuracy': 100.0 * accuracy_score(targets, predictions),
        'macro_precision': 100.0 * precision_score(
            targets, predictions, average='macro', zero_division=0),
        'macro_recall': 100.0 * recall_score(
            targets, predictions, average='macro', zero_division=0),
        'macro_f1': 100.0 * f1_score(
            targets, predictions, average='macro', zero_division=0),
        'weighted_f1': 100.0 * f1_score(
            targets, predictions, average='weighted', zero_division=0),
    }
    if return_predictions:
        return (metrics, targets, predictions, np.asarray(indices),
                np.concatenate(probabilities, axis=0))
    return metrics


def save_model(model, path: Path, split_seed: int, classes, phase: str,
               epoch: int, validation_accuracy: float):
    temp_path = path.with_suffix('.tmp')
    torch.save({
        'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'model_name': MODEL_NAME,
        'classes': classes,
        'split_seed': split_seed,
        'training_seed': TRAINING_SEED,
        'phase': phase,
        'epoch': epoch,
        'best_val_accuracy': validation_accuracy,
    }, temp_path)
    os.replace(temp_path, path)


def run_split(args, split_seed, paths, labels, classes, images,
              train_transform, eval_transform, device):
    output = args.output_dir / f'seed_{split_seed}'
    output.mkdir(parents=True, exist_ok=True)
    final_metrics_path = output / 'final_metrics.json'
    if args.resume and final_metrics_path.exists() and (output / 'test_predictions.csv').exists():
        print(f'[SKIP] Completed split {split_seed}', flush=True)
        return json.loads(final_metrics_path.read_text())

    set_seed(TRAINING_SEED)
    train_idx, val_idx, test_idx = make_split(labels, split_seed)
    (output / 'split_indices.json').write_text(json.dumps({
        'split_seed': split_seed, 'training_seed': TRAINING_SEED,
        'train': train_idx.tolist(), 'val': val_idx.tolist(),
        'test': test_idx.tolist(), 'class_names': classes,
        'paths': [str(path) for path in paths],
    }, indent=2))

    train_loader = make_loader(
        RAMDataset(images, labels, train_idx, train_transform),
        True, args.batch_size, args.workers, args.prefetch)
    val_loader = make_loader(
        RAMDataset(images, labels, val_idx, eval_transform),
        False, args.batch_size, args.eval_workers, args.prefetch)
    test_loader = make_loader(
        RAMDataset(images, labels, test_idx, eval_transform),
        False, args.batch_size, args.eval_workers, args.prefetch)

    model = create_model(len(classes), not args.no_pretrained).to(
        device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    print(f'\n[SPLIT {split_seed}] train={len(train_idx)} val={len(val_idx)} '
          f'test={len(test_idx)}', flush=True)

    # Stage 1: frozen backbone; head training over five epochs.
    freeze_backbone_train_head(model)
    optimizer = torch.optim.AdamW(build_simple_param_groups(model, PHASE1_LR))
    scheduler = create_scheduler(optimizer, args.head_epochs)
    best_head_accuracy = float('-inf')
    for epoch in range(1, args.head_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.accum)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({
            'phase': 'phase1_head_only', 'epoch': epoch,
            'train_loss': train_loss, 'train_accuracy': train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_precision': val_metrics['macro_precision'],
            'val_macro_recall': val_metrics['macro_recall'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_seconds': time.time() - t0,
        })
        pd.DataFrame(history).to_csv(output / 'training_log.csv', index=False)
        if val_metrics['accuracy'] > best_head_accuracy:
            best_head_accuracy = val_metrics['accuracy']
            save_model(model, output / 'best_head_model.pt', split_seed, classes,
                       'phase1_head_only', epoch, best_head_accuracy)
        print(f'  Stage 1 {epoch:02d}/{args.head_epochs}: '
              f'val={val_metrics["accuracy"]:.3f}% '
              f'lr={optimizer.param_groups[0]["lr"]:.3e}', flush=True)
    del optimizer, scheduler

    # Stage 2 continues directly from final Stage 1 state, not head best state.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(build_layer_decay_param_groups(model))
    scheduler = create_scheduler(optimizer, args.finetune_epochs,
                                 PHASE2_WARMUP_EPOCHS)
    best_val_accuracy = float('-inf')
    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.accum)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({
            'phase': 'phase2_full_finetune', 'epoch': epoch,
            'train_loss': train_loss, 'train_accuracy': train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_precision': val_metrics['macro_precision'],
            'val_macro_recall': val_metrics['macro_recall'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_seconds': time.time() - t0,
        })
        pd.DataFrame(history).to_csv(output / 'training_log.csv', index=False)
        # Highest validation accuracy; later epoch replaces earlier ties.
        if val_metrics['accuracy'] >= best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_model(model, output / 'best_model.pt', split_seed, classes,
                       'phase2_full_finetune', epoch, best_val_accuracy)
        print(f'  Stage 2 {epoch:02d}/{args.finetune_epochs}: '
              f'val={val_metrics["accuracy"]:.3f}% '
              f'lr={optimizer.param_groups[0]["lr"]:.3e}', flush=True)

    checkpoint = torch.load(output / 'best_model.pt', map_location='cpu',
                            weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    metrics, truth, predicted, source_indices, probs = evaluate(
        model, test_loader, criterion, device, return_predictions=True)
    result = {
        'split_seed': split_seed,
        'training_seed': TRAINING_SEED,
        'model': MODEL_NAME,
        'best_stage2_epoch': int(checkpoint['epoch']),
        'best_validation_accuracy': float(best_val_accuracy),
        'train_count': int(len(train_idx)),
        'val_count': int(len(val_idx)),
        'test_count': int(len(test_idx)),
        'test_accuracy': float(metrics['accuracy']),
        'test_macro_f1': float(metrics['macro_f1']),
        'test_weighted_f1': float(metrics['weighted_f1']),
        'test_loss': float(metrics['loss']),
    }

    prediction_columns = {
        'image_path': [str(paths[i]) for i in source_indices],
        'true_class': [classes[i] for i in truth],
        'predicted_class': [classes[i] for i in predicted],
        'confidence': probs.max(axis=1).tolist(),
        'correct': (truth == predicted).tolist(),
    }
    for class_index, class_name in enumerate(classes):
        prediction_columns['prob_' + class_name.lower().replace(' ', '_')] = (
            probs[:, class_index].tolist())
    pd.DataFrame(prediction_columns).to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame(
        confusion_matrix(truth, predicted, labels=list(range(len(classes)))),
        index=classes, columns=classes,
    ).to_csv(output / 'confusion_matrix.csv')
    pd.DataFrame(classification_report(
        truth, predicted, labels=list(range(len(classes))),
        target_names=classes, zero_division=0, output_dict=True,
    )).T.to_csv(output / 'classification_report.csv')
    final_metrics_path.write_text(json.dumps(result, indent=2))
    print(f'[SPLIT {split_seed}] TEST {metrics["accuracy"]:.4f}% '
          f'best Stage 2 epoch {checkpoint["epoch"]}', flush=True)

    del model, optimizer, scheduler, scaler, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required.')
    if args.accum <= 0 or args.batch_size <= 0:
        raise ValueError('--batch-size and --accum must be positive.')
    if args.head_epochs <= 0 or args.finetune_epochs <= PHASE2_WARMUP_EPOCHS:
        raise ValueError('Invalid epoch counts for the configured schedulers.')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = find_data_root(args.data_dir)
    paths, labels, classes = discover_dataset(root)
    print('Device:', torch.cuda.get_device_name(0))
    print('Classes:', classes)
    print('Split seeds:', args.seeds, '| Training seed:', TRAINING_SEED)
    print('Batch:', args.batch_size, '| Accum:', args.accum)
    images = predecode_to_ram(paths)
    train_transform, eval_transform = build_transforms()

    results = []
    for split_seed in args.seeds:
        results.append(run_split(args, split_seed, paths, labels, classes, images,
                                 train_transform, eval_transform, device))
        pd.DataFrame(results).sort_values('split_seed').to_csv(
            args.output_dir / 'seed_summary.csv', index=False)
    print('Summary saved:', args.output_dir / 'seed_summary.csv')


if __name__ == '__main__':
    main()
