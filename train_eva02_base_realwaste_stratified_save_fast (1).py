#!/usr/bin/env python3
"""RealWaste EVA02-Base two-stage fine-tuning with per-seed artifacts.

Requires: torch, torchvision, timm, scikit-learn, numpy, pandas, pillow.
NOTE: check AUGMENTATIONS and preprocessing against your previous RealWaste
script before calling a run an exact reproduction.
"""
import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
TRAIN_SEED = 42
COUNTS = (3326, 617, 809)
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=None,
                   help='Root containing nine class directories (or a parent directory).')
    p.add_argument('--output-dir', type=Path, default=Path.home() / 'realwaste_eva02_stratified_saved')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--eval-workers', type=int, default=2)
    p.add_argument('--prefetch', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--head-epochs', type=int, default=5)
    p.add_argument('--finetune-epochs', type=int, default=30)
    p.add_argument('--resume', action='store_true', help='Skip seeds with final_metrics.json (not partial epoch resume).')
    p.add_argument('--no-pretrained', action='store_true', help='Only for local testing; normally do not use.')
    return p.parse_args()


def find_data_root(requested):
    candidates = []
    if requested: candidates.append(requested.expanduser())
    else:
        base = Path.home()
        candidates += [base / x for x in ('RealWaste', 'realwaste', 'RealWaste-main', 'datasets/RealWaste',
                                           'datasets/realwaste', 'data/RealWaste', 'realwaste_dataset')]
    expected = {'Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',
                'Paper', 'Plastic', 'Textile Trash', 'Vegetation'}
    for root in candidates:
        if not root.exists(): continue
        for folder in [root] + [x for x in root.iterdir() if x.is_dir()]:
            if expected.issubset({x.name for x in folder.iterdir() if x.is_dir()}):
                return folder
    raise FileNotFoundError('Could not locate nine RealWaste class folders. Supply --data-dir /path/to/RealWaste')


def collect(root):
    classes = sorted(x.name for x in root.iterdir() if x.is_dir())
    assert len(classes) == 9, f'Expected 9 folders, found {classes}'
    paths, y = [], []
    for i, cls in enumerate(classes):
        for file in sorted((root / cls).rglob('*')):
            if file.is_file() and file.suffix.lower() in EXT:
                paths.append(file); y.append(i)
    assert len(paths) == 4752, f'Expected 4752 images; found {len(paths)}'
    print('Classes:', classes, flush=True)
    print('Images:', len(paths), 'Counts:', {c:y.count(i) for i,c in enumerate(classes)}, flush=True)
    return paths, np.array(y, dtype=np.int64), classes


def decode(paths):
    print('Predecoding all 4752 RGB images to system RAM...', flush=True)
    arrays = []
    nbytes = 0
    t0 = time.time()
    for n, p in enumerate(paths, 1):
        with Image.open(p) as im:
            arr = np.asarray(im.convert('RGB')).copy()
        arrays.append(arr)
        nbytes += arr.nbytes
        if n % 500 == 0:
            print(f'  {n}/{len(paths)} images; decoded RAM {nbytes / 2**30:.2f} GiB', flush=True)
    print(f'RGB cache: {nbytes / 2**30:.2f} GiB in {time.time()-t0:.1f}s', flush=True)
    return arrays


# Replace with transforms from the original script if exact reproduction is required.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(448, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)
])
EVAL_TF = transforms.Compose([
    transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)
])


class RAMDataset(Dataset):
    def __init__(self, arrays, labels, indices, transform):
        self.arrays = arrays; self.labels = labels; self.indices = np.asarray(indices); self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        j = int(self.indices[idx]); img = Image.fromarray(self.arrays[j])
        return self.transform(img), int(self.labels[j]), j


def loader(ds, batch, workers, shuffle, prefetch):
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=workers,
                  pin_memory=True, drop_last=False, persistent_workers=workers > 0)
    if workers: kwargs['prefetch_factor'] = prefetch
    return DataLoader(ds, **kwargs)


def split_seed(y, s):
    all_ids = np.arange(len(y))
    tr, rem = train_test_split(all_ids, train_size=COUNTS[0], random_state=s, stratify=y)
    va, te = train_test_split(rem, train_size=COUNTS[1], random_state=s,
                              stratify=y[rem])
    assert (len(tr), len(va), len(te)) == COUNTS
    assert len(set(tr) & set(te)) == 0
    return tr, va, te


def make_model(num_classes, pretrained):
    m = timm.create_model(MODEL, pretrained=pretrained, num_classes=num_classes)
    return m


def head_params(model):
    names = ('head', 'fc', 'classifier')
    params = [p for n,p in model.named_parameters() if p.requires_grad and
              (n.split('.')[0] in names)]
    if not params: raise RuntimeError('Cannot identify classifier head parameters')
    return params


def set_head_only(model):
    for p in model.parameters(): p.requires_grad_(False)
    for p in head_params_unfrozen(model): p.requires_grad_(True)


def head_params_unfrozen(model):
    return [p for n,p in model.named_parameters() if n.split('.')[0] in ('head','fc','classifier')]


def layer_id(name, depth):
    if name.startswith(('head.', 'fc.', 'classifier.')): return depth + 1
    if name.startswith('blocks.'):
        parts = name.split('.')
        if len(parts) > 1 and parts[1].isdigit(): return int(parts[1]) + 1
    return 0


def layerwise_groups(model, backbone_lr=8e-6, head_lr=4e-5, decay=0.75, weight_decay=1e-4):
    depth = len(model.blocks)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        lid = layer_id(name, depth)
        is_head = lid == depth + 1
        lr = head_lr if is_head else backbone_lr * decay ** (depth - lid)
        no_decay = param.ndim <= 1 or name.endswith('.bias') or 'pos_embed' in name or 'cls_token' in name
        wd = 0 if no_decay else weight_decay
        key = (lid, wd)
        if key not in groups: groups[key] = {'params':[], 'lr':lr, 'weight_decay':wd, 'initial_lr':lr}
        groups[key]['params'].append(param)
    return list(groups.values())


def amp_ctx():
    return torch.autocast(device_type='cuda', dtype=torch.float16)


def train_epoch(model, dl, optim, scaler, device, accum, epoch, epochs, mode, warmup_epochs=0):
    model.train()
    optim.zero_grad(set_to_none=True)
    steps = (len(dl) + accum - 1)//accum
    # Keep epoch statistics on the GPU: avoids two forced CUDA synchronizations
    # per batch caused by Tensor.item(). Synchronize once at epoch end.
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    start = time.time()
    for i, (x,y,_) in enumerate(dl):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if mode == 'finetune':
            t = epoch + (i // accum)/max(1,steps)
            mult = (t + 1e-8)/max(1, warmup_epochs) if t < warmup_epochs else \
                0.5 * (1 + np.cos(np.pi * (t-warmup_epochs)/max(1,epochs-warmup_epochs)))
            for pg in optim.param_groups: pg['lr'] = pg['initial_lr'] * float(mult)
        # Correct scaling also for an incomplete last accumulation group.
        group_start = (i // accum)*accum
        group_len = min(accum, len(dl)-group_start)
        with amp_ctx():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.05)
        scaler.scale(loss / group_len).backward()
        if (i+1) % accum == 0 or i+1 == len(dl):
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            loss_sum += loss.detach().to(torch.float64) * y.numel()
            correct += (logits.detach().argmax(1) == y).sum()
        total += y.numel()
        if (i+1) % 100 == 0:
            print(f'    {mode} epoch {epoch+1}/{epochs}: batch {i+1}/{len(dl)}, '
                  f'GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    return {'train_loss':loss_sum.item()/total, 'train_acc':correct.item()/total*100,
            'train_seconds':elapsed, 'train_images_per_sec':total/elapsed}


@torch.inference_mode()
def evaluate(model, dl, device, return_predictions=False):
    model.eval()
    ys, preds, ids, probs = [], [], [], []
    total_loss = 0.; total = 0
    for x,y,idx in dl:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        yy = y.to(device, non_blocking=True)
        with amp_ctx(): logits = model(x)
        total_loss += nn.functional.cross_entropy(logits.float(), yy, reduction='sum').item()
        pr = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(pr)
        ys.extend(y.tolist()); preds.extend(pr.argmax(axis=1).tolist()); ids.extend(idx.tolist())
        total += len(y)
    ys = np.asarray(ys); preds=np.asarray(preds)
    d = {'loss':total_loss/total, 'accuracy':100*accuracy_score(ys,preds),
         'macro_f1':100*f1_score(ys,preds,average='macro',zero_division=0),
         'weighted_f1':100*f1_score(ys,preds,average='weighted',zero_division=0)}
    if return_predictions: return d, ys, preds, np.asarray(ids), np.concatenate(probs)
    return d


def atomic_save(model, path, seed, classes, stage, epoch):
    tmp = path.with_suffix('.tmp')
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'model_name':MODEL, 'classes':classes, 'split_seed':seed,
                'training_seed':TRAIN_SEED, 'stage':stage, 'epoch':epoch}, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('CUDA device:', torch.cuda.get_device_name(0), flush=True)
    root = find_data_root(args.data_dir)
    paths,y,classes = collect(root)
    arrays = decode(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device('cuda')
    for seed in args.seeds:
        out=args.output_dir / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        if args.resume and (out/'final_metrics.json').exists():
            print(f'Skipping completed seed {seed}', flush=True); continue
        seed_all(TRAIN_SEED)
        tr,va,te=split_seed(y,seed)
        (out/'split_indices.json').write_text(json.dumps({
            'split_seed':seed,'training_seed':TRAIN_SEED,
            'train':tr.tolist(),'val':va.tolist(),'test':te.tolist(),
            'class_names':classes,'paths':[str(x) for x in paths]},indent=2))
        # fresh loaders per seed; persistent workers speed subsequent epochs.
        train_dl=loader(RAMDataset(arrays,y,tr,TRAIN_TF),args.batch_size,args.workers,True,args.prefetch)
        val_dl=loader(RAMDataset(arrays,y,va,EVAL_TF),args.batch_size,args.eval_workers,False,args.prefetch)
        test_dl=loader(RAMDataset(arrays,y,te,EVAL_TF),args.batch_size,args.eval_workers,False,args.prefetch)
        print(f'\n====== SEED {seed}: train={len(tr)} val={len(va)} test={len(te)} ======',flush=True)
        model=make_model(len(classes),not args.no_pretrained).to(device,memory_format=torch.channels_last)
        set_head_only(model)
        optimizer=torch.optim.AdamW(head_params(model),lr=1e-3,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        log=[]
        best_head=-1.
        # Stage 1 trains classifier only.
        for ep in range(args.head_epochs):
            td=train_epoch(model,train_dl,optimizer,scaler,device,args.accum,ep,args.head_epochs,'head')
            vd=evaluate(model,val_dl,device)
            log.append({'stage':'head','epoch':ep+1,**td,**{'val_'+k:v for k,v in vd.items()}})
            pd.DataFrame(log).to_csv(out/'training_log.csv',index=False)
            print('HEAD',ep+1,'val acc',round(vd['accuracy'],4),'val macroF1',round(vd['macro_f1'],4),flush=True)
            if vd['accuracy'] > best_head:
                best_head=vd['accuracy']
                atomic_save(model,out/'best_head_model.pt',seed,classes,'head',ep+1)
        del optimizer,scaler
        gc.collect(); torch.cuda.empty_cache()
        # Stage 2 begins from best stage-1 model.
        head_ckpt=torch.load(out/'best_head_model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(head_ckpt['model']); del head_ckpt
        for p in model.parameters(): p.requires_grad_(True)
        if hasattr(model,'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        optimizer=torch.optim.AdamW(layerwise_groups(model),weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        best=-1.
        for ep in range(args.finetune_epochs):
            td=train_epoch(model,train_dl,optimizer,scaler,device,args.accum,ep,
                           args.finetune_epochs,'finetune',warmup_epochs=2)
            vd=evaluate(model,val_dl,device)
            log.append({'stage':'finetune','epoch':ep+1,**td,
                        **{'val_'+k:v for k,v in vd.items()}})
            pd.DataFrame(log).to_csv(out/'training_log.csv',index=False)
            print('FINETUNE',ep+1,'val acc',round(vd['accuracy'],4),
                  'val macroF1',round(vd['macro_f1'],4),'train seconds',round(td['train_seconds']),flush=True)
            # checkpoint based solely on validation, never test.
            if vd['accuracy'] > best:
                best=vd['accuracy']
                atomic_save(model,out/'best_model.pt',seed,classes,'finetune',ep+1)
        best_ckpt=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_ckpt['model'])
        result,truth,pred,indices,prob=evaluate(model,test_dl,device,True)
        d={
            'split_seed':seed,'training_seed':TRAIN_SEED,'model':MODEL,
            'best_stage2_epoch':best_ckpt['epoch'],'best_validation_accuracy':best,
            'train_count':len(tr),'val_count':len(va),'test_count':len(te),
            'test_accuracy':result['accuracy'],'test_macro_f1':result['macro_f1'],
            'test_weighted_f1':result['weighted_f1'],'test_loss':result['loss']
        }
        data={'image_path':[str(paths[i]) for i in indices],
              'true_class':[classes[i] for i in truth],
              'predicted_class':[classes[i] for i in pred],
              'confidence':prob.max(axis=1).tolist(),
              'correct':(truth == pred).tolist()}
        for j,c in enumerate(classes):
            data['prob_'+c.lower().replace(' ','_')]=prob[:,j].tolist()
        pd.DataFrame(data).to_csv(out/'test_predictions.csv',index=False)
        pd.DataFrame(confusion_matrix(truth,pred,labels=list(range(len(classes)))),
                     index=classes,columns=classes).to_csv(out/'confusion_matrix.csv')
        report=classification_report(truth,pred,target_names=classes,
                                     labels=list(range(len(classes))),zero_division=0,output_dict=True)
        pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
        (out/'final_metrics.json').write_text(json.dumps(d,indent=2))
        print('SEED',seed,'FINAL:',json.dumps(d,indent=2),flush=True)
        del model, optimizer, scaler, train_dl, val_dl, test_dl, best_ckpt
        gc.collect(); torch.cuda.empty_cache()
    summaries=[]
    for seed in args.seeds:
        path=args.output_dir/f'seed_{seed}'/'final_metrics.json'
        if path.exists(): summaries.append(json.loads(path.read_text()))
    if summaries:
        pd.DataFrame(summaries).to_csv(args.output_dir/'seed_summary.csv',index=False)
        print('Summary saved:',args.output_dir/'seed_summary.csv',flush=True)


if __name__ == '__main__': main()
