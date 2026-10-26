#!/usr/bin/env python3
import os
LOCAL_SCRATCH = os.environ.get('TMPDIR', '/tmp')
os.environ.setdefault('HF_HOME', os.path.join(LOCAL_SCRATCH, 'hf_home'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(LOCAL_SCRATCH, 'hf_home', 'hub'))
os.environ.setdefault('TORCH_HOME', os.path.join(LOCAL_SCRATCH, 'torch_home'))

import argparse, gc, json, math, random, shutil, sys, time, io
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from safetensors.torch import load_file as load_safetensors

MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
HF_REPO = 'timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
SPLIT_SEEDS = [0,1,2,3,4]
TRAINING_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.13, 0.17
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
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
VALID_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}
DATA_CONFIG = {
    'input_size': (3,448,448),
    'mean': (0.48145466,0.4578275,0.40821073),
    'std': (0.26862954,0.26130258,0.27577711),
    'interpolation':'bicubic',
    'crop_pct':1.0,
    'crop_mode':'squash',
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    s = TRAINING_SEED + worker_id
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def resolve_data_root(raw):
    for p in [Path(raw), Path.cwd()/raw, Path.home()/raw, Path('/home/user/22104053')/raw]:
        p = p.expanduser().resolve()
        if p.exists() and p.is_dir(): return p
    raise FileNotFoundError(f'Could not find dataset root {raw}. Expected trashnet/dataset-resized/cardboard/...')

def discover_dataset(root):
    expected={'cardboard','glass','metal','paper','plastic','trash'}
    if not expected.issubset({p.name.lower() for p in root.iterdir() if p.is_dir()}):
        for sub in ('dataset-resized','dataset_resized','trashnet','TrashNet'):
            candidate=root/sub
            if candidate.is_dir() and expected.issubset({p.name.lower() for p in candidate.iterdir() if p.is_dir()}):
                root=candidate; break
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    rows=[]
    for c in classes:
        for p in sorted((root/c).rglob('*')):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                rows.append({'path':str(p),'label':class_to_idx[c],'class':c})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No images found.')
    if len(df)!=2527 or len(classes)!=6: raise RuntimeError(f'Expected 2527 images in 6 classes, found {len(df)} images in {len(classes)} classes. Check --data.')
    return df, classes

def cache_compressed_bytes(df):
    # Unlike RealWaste, avoid storing original decoded RGB arrays in RAM.
    print('CACHING COMPRESSED TRASHNET IMAGE BYTES INTO RAM',flush=True)
    images=[]; total=0; t0=time.time()
    for i,path in enumerate(df['path'].tolist(),1):
        with open(path,'rb') as f: b=f.read()
        images.append(b); total+=len(b)
        if i%500==0 or i==len(df):
            print(f'Cached {i}/{len(df)} | compressed RAM {total/1024**3:.2f} GiB | {time.time()-t0:.1f}s',flush=True)
    return images

def make_split(df, seed):
    idx=np.arange(len(df)); y=df['label'].to_numpy()
    train,temp=train_test_split(idx,test_size=VAL_RATIO+TEST_RATIO,random_state=seed,shuffle=True,stratify=y)
    val,test=train_test_split(temp,test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),random_state=seed,shuffle=True,stratify=y[temp])
    return np.asarray(train),np.asarray(val),np.asarray(test)

def build_transforms():
    train=create_transform(input_size=DATA_CONFIG['input_size'],is_training=True,mean=DATA_CONFIG['mean'],std=DATA_CONFIG['std'],interpolation=DATA_CONFIG['interpolation'],hflip=0.5,color_jitter=0.08,re_prob=0.10,re_mode='pixel',re_count=1)
    eval_t=create_transform(input_size=DATA_CONFIG['input_size'],is_training=False,mean=DATA_CONFIG['mean'],std=DATA_CONFIG['std'],interpolation=DATA_CONFIG['interpolation'],crop_pct=DATA_CONFIG['crop_pct'])
    return train,eval_t

class CachedDataset(Dataset):
    def __init__(self, decoded, labels, indices, transform):
        self.decoded=decoded; self.labels=labels; self.indices=np.asarray(indices); self.transform=transform
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j=int(self.indices[i])
        with Image.open(io.BytesIO(self.decoded[j])) as raw:
            img=raw.convert('RGB')
        return self.transform(img), int(self.labels[j])

def make_loader(ds, train, workers, prefetch, persistent=True):
    g=torch.Generator(); g.manual_seed(TRAINING_SEED)
    kw=dict(dataset=ds,batch_size=BATCH_SIZE,shuffle=train,drop_last=False,num_workers=workers,pin_memory=True,
            worker_init_fn=seed_worker if workers>0 else None,persistent_workers=(persistent if workers>0 else False),
            generator=(g if train else None))
    if workers>0: kw['prefetch_factor']=prefetch
    return DataLoader(**kw)

def get_pretrained_path():
    d=Path(LOCAL_SCRATCH)/f'eva02_base_weights_{os.getuid()}'; d.mkdir(parents=True,exist_ok=True)
    target=d/'model.safetensors'
    if target.exists(): return target
    for src in [Path.home()/'pretrained/eva02_base_448/model.safetensors', Path('/home/user/22104053/pretrained/eva02_base_448/model.safetensors')]:
        if src.exists():
            print(f'[MODEL] Copying weights from {src} to /tmp')
            shutil.copy2(src,target); return target
    from huggingface_hub import hf_hub_download
    print('[MODEL] Downloading pretrained EVA02-Base weights to /tmp')
    return Path(hf_hub_download(repo_id=HF_REPO,filename='model.safetensors',local_dir=str(d)))

def clean_keys(k):
    out=[k]
    for prefix in ['module.','_orig_mod.','model.']:
        if k.startswith(prefix): out.append(k[len(prefix):])
    return out

def create_model(nclasses, pretrained_path):
    m=timm.create_model(MODEL_NAME,pretrained=False,num_classes=nclasses)
    if USE_GRAD_CHECKPOINTING and hasattr(m,'set_grad_checkpointing'): m.set_grad_checkpointing(enable=True)
    state=load_safetensors(str(pretrained_path)); ms=m.state_dict(); filt={}; loaded=skipped=0
    for k,v in state.items():
        ok=False
        for ck in clean_keys(k):
            if ck in ms and v.shape==ms[ck].shape:
                filt[ck]=v; loaded+=1; ok=True; break
        if not ok: skipped+=1
    missing,unexpected=m.load_state_dict(filt,strict=False)
    print(f'[MODEL] loaded={loaded} skipped={skipped} missing={len(missing)} unexpected={len(unexpected)}')
    return m

def freeze_backbone_train_head(m):
    for p in m.parameters(): p.requires_grad=False
    if hasattr(m,'head'):
        for p in m.head.parameters(): p.requires_grad=True
    else:
        for n,p in m.named_parameters():
            if 'head' in n or 'classifier' in n or 'fc' in n: p.requires_grad=True

def unfreeze_all(m):
    for p in m.parameters(): p.requires_grad=True

def get_layer_id(name,nblocks):
    if name.startswith('head'): return nblocks+2
    if name.startswith('norm'): return nblocks+1
    if name.startswith('blocks.'):
        parts=name.split('.')
        if len(parts)>1 and parts[1].isdigit(): return int(parts[1])+1
    return 0

def build_simple_groups(m,lr):
    decay=[]; no=[]
    for n,p in m.named_parameters():
        if not p.requires_grad: continue
        (no if (p.ndim<=1 or n.endswith('.bias') or 'pos_embed' in n or 'cls_token' in n) else decay).append(p)
    return [{'params':decay,'lr':lr,'weight_decay':WEIGHT_DECAY},{'params':no,'lr':lr,'weight_decay':0.0}]

def build_layer_groups(m):
    nblocks=len(m.blocks) if hasattr(m,'blocks') else 12; maxid=nblocks+2; groups={}
    for n,p in m.named_parameters():
        if not p.requires_grad: continue
        no=(p.ndim<=1 or n.endswith('.bias') or 'pos_embed' in n or 'cls_token' in n)
        wd=0.0 if no else WEIGHT_DECAY
        if n.startswith('head'): lr=PHASE2_HEAD_LR
        else: lr=PHASE2_BASE_LR*(LAYER_DECAY**(maxid-get_layer_id(n,nblocks)))
        groups.setdefault((lr,wd),{'params':[],'lr':lr,'weight_decay':wd})['params'].append(p)
    return list(groups.values())

def create_scheduler(opt,epochs,warmup=0):
    if warmup<=0: return optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    w=optim.lr_scheduler.LinearLR(opt,start_factor=0.10,end_factor=1.0,total_iters=warmup)
    c=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs-warmup))
    return optim.lr_scheduler.SequentialLR(opt,schedulers=[w,c],milestones=[warmup])

def move_batch(x,y,device):
    x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
    if USE_CHANNELS_LAST: x=x.contiguous(memory_format=torch.channels_last)
    return x,y

def train_epoch(m,loader,opt,crit,scaler,device):
    m.train(); opt.zero_grad(set_to_none=True); loss_sum=0.0; correct=total=0
    for bi,(x,y) in enumerate(loader,1):
        x,y=move_batch(x,y,device)
        with torch.amp.autocast('cuda',dtype=AMP_DTYPE,enabled=True):
            out=m(x); loss=crit(out,y)
        scaler.scale(loss/ACCUMULATION_STEPS).backward()
        if bi%ACCUMULATION_STEPS==0 or bi==len(loader):
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        pred=out.argmax(1); correct += (pred==y).sum().item(); total += y.size(0); loss_sum += loss.item()*y.size(0)
    return loss_sum/total, 100.0*correct/total

@torch.no_grad()
def evaluate(m,loader,crit,device):
    m.eval(); loss_sum=0.0; total=0; ys=[]; ps=[]
    for x,y in loader:
        x,y=move_batch(x,y,device)
        with torch.amp.autocast('cuda',dtype=AMP_DTYPE,enabled=True):
            out=m(x); loss=crit(out,y)
        p=out.argmax(1); loss_sum += loss.item()*y.size(0); total += y.size(0); ys.extend(y.cpu().tolist()); ps.extend(p.cpu().tolist())
    yt=np.asarray(ys); yp=np.asarray(ps)
    met={'accuracy':100*accuracy_score(yt,yp),'macro_precision':100*precision_score(yt,yp,average='macro',zero_division=0),
         'macro_recall':100*recall_score(yt,yp,average='macro',zero_division=0),'macro_f1':100*f1_score(yt,yp,average='macro',zero_division=0),
         'weighted_f1':100*f1_score(yt,yp,average='weighted',zero_division=0)}
    return loss_sum/total,met,yt,yp


@torch.no_grad()
def save_test_predictions(model, loader, device, df, test_idx, classes, destination):
    # Deterministic evaluation with exactly the original eval transform and test index order.
    model.eval()
    probabilities=[]
    for x,y in loader:
        x,y=move_batch(x,y,device)
        with torch.amp.autocast('cuda',dtype=AMP_DTYPE,enabled=True):
            logits=model(x)
        probabilities.append(torch.softmax(logits.float(),dim=1).cpu().numpy())
    probs=np.concatenate(probabilities,axis=0)
    true=df['label'].to_numpy(dtype=np.int64)[test_idx]
    predicted=np.argmax(probs,axis=1)
    output=pd.DataFrame({
        'image_path': df.iloc[test_idx]['path'].to_numpy(),
        'true_class': [classes[int(i)] for i in true],
        'predicted_class': [classes[int(i)] for i in predicted],
        'confidence': probs.max(axis=1),
        'correct': predicted==true
    })
    for k,name in enumerate(classes):
        output['prob_'+name.lower().replace(' ','_')]=probs[:,k]
    output.to_csv(destination,index=False)
    print(f'[SAVED] {len(output)} test predictions -> {destination}',flush=True)

def run_seed(seed,df,decoded,classes,train_t,eval_t,weights,outroot,device,workers,eval_workers,prefetch):
    seed_dir=outroot/f'seed_{seed}'; seed_dir.mkdir(parents=True,exist_ok=True)
    final=seed_dir/'final_metrics.json'
    if final.exists() and (seed_dir/'best_model.pt').exists() and (seed_dir/'test_predictions.csv').exists():
        print(f'[SKIP] seed {seed} already complete with checkpoint and predictions'); return json.loads(final.read_text())
    train_idx,val_idx,test_idx=make_split(df,seed)
    print('\n'+'='*100); print(f'SPLIT SEED {seed}'); print('='*100); print(f'Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}')
    (seed_dir/'split_indices.json').write_text(json.dumps({'split_seed':seed,'training_seed':TRAINING_SEED,'train_indices':train_idx.tolist(),'val_indices':val_idx.tolist(),'test_indices':test_idx.tolist(),'class_names':classes}))
    labels=df['label'].to_numpy(dtype=np.int64)
    tr_ds=CachedDataset(decoded,labels,train_idx,train_t); va_ds=CachedDataset(decoded,labels,val_idx,eval_t); te_ds=CachedDataset(decoded,labels,test_idx,eval_t)
    tr=make_loader(tr_ds,True,workers,prefetch,True); va=make_loader(va_ds,False,eval_workers,min(prefetch,4),True); te=make_loader(te_ds,False,eval_workers,min(prefetch,4),False)
    set_seed(TRAINING_SEED)
    m=create_model(len(classes),weights).to(device)
    if USE_CHANNELS_LAST: m=m.to(memory_format=torch.channels_last)
    crit=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING); scaler=torch.amp.GradScaler('cuda',enabled=True)
    tmp=Path(LOCAL_SCRATCH)/f'eva02_trashnet_seed_{seed}_{os.getuid()}'; tmp.mkdir(parents=True,exist_ok=True); best=tmp/'best_model.pt'
    phases=[('phase1_head_only',PHASE1_EPOCHS,True,0),('phase2_full_finetune',PHASE2_EPOCHS,False,PHASE2_WARMUP_EPOCHS)]
    best_acc=0.0; hist=[]; seed_start=time.time()
    for phase_i,(name,epochs,freeze,warmup) in enumerate(phases):
        print('\n'+name); print('-'*100)
        if freeze: freeze_backbone_train_head(m); groups=build_simple_groups(m,PHASE1_LR)
        else: unfreeze_all(m); groups=build_layer_groups(m)
        opt=optim.AdamW(groups); sched=create_scheduler(opt,epochs,warmup)
        for ep in range(1,epochs+1):
            t0=time.time(); tl,ta=train_epoch(m,tr,opt,crit,scaler,device); vl,vm,_,_=evaluate(m,va,crit,device); sched.step(); sec=time.time()-t0; lr=opt.param_groups[0]['lr']
            print(f'[Seed {seed}] {name} {ep:02d}/{epochs} | train loss {tl:.4f} | train acc {ta:.2f}% | val loss {vl:.4f} | val acc {vm["accuracy"]:.2f}% | macro F1 {vm["macro_f1"]:.2f}% | LR {lr:.2e} | {sec:.1f}s',flush=True)
            hist.append({'phase':name,'epoch':ep,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':vm['accuracy'],'val_macro_precision':vm['macro_precision'],'val_macro_recall':vm['macro_recall'],'val_macro_f1':vm['macro_f1'],'val_weighted_f1':vm['weighted_f1'],'lr':lr,'epoch_seconds':sec})
            if vm['accuracy'] >= best_acc:
                best_acc=vm['accuracy']; torch.save({'model_state_dict':m.state_dict(),'epoch':ep,'phase':name,'best_val_accuracy':best_acc,'best_val_loss':vl},best)
                print(f'  -> BEST validation accuracy {best_acc:.2f}%',flush=True)
    ck=torch.load(best,map_location=device,weights_only=False); m.load_state_dict(ck['model_state_dict'],strict=True)
    shutil.copy2(best,seed_dir/'best_model.pt')
    print(f'[SAVED] Best model | {ck["phase"]} epoch {ck["epoch"]} | validation accuracy {ck["best_val_accuracy"]:.4f}% | {seed_dir/"best_model.pt"}',flush=True)
    test_loss,tm,yt,yp=evaluate(m,te,crit,device)
    save_test_predictions(m,te,device,df,test_idx,classes,seed_dir/'test_predictions.csv')
    result={'split_seed':seed,'accuracy':float(tm['accuracy']),'macro_precision':float(tm['macro_precision']),'macro_recall':float(tm['macro_recall']),'macro_f1':float(tm['macro_f1']),'weighted_f1':float(tm['weighted_f1']),'test_loss':float(test_loss),'best_val_accuracy':float(ck['best_val_accuracy']),'best_val_loss':float(ck['best_val_loss']),'best_epoch':int(ck['epoch']),'best_phase':ck['phase'],'train_size':len(train_idx),'val_size':len(val_idx),'test_size':len(test_idx),'seed_seconds':time.time()-seed_start}
    final.write_text(json.dumps(result,indent=2)); pd.DataFrame(hist).to_csv(seed_dir/'training_log.csv',index=False)
    cm=confusion_matrix(yt,yp,labels=list(range(len(classes)))); pd.DataFrame(cm,index=classes,columns=classes).to_csv(seed_dir/'confusion_matrix.csv')
    pd.DataFrame(classification_report(yt,yp,target_names=classes,output_dict=True,zero_division=0)).T.to_csv(seed_dir/'classification_report.csv')
    print('\n'+'='*100); print(f'SEED {seed} TEST RESULT'); print('='*100)
    print(f'Accuracy         : {result["accuracy"]:.4f}%'); print(f'Macro precision  : {result["macro_precision"]:.4f}%'); print(f'Macro recall     : {result["macro_recall"]:.4f}%'); print(f'Macro F1         : {result["macro_f1"]:.4f}%'); print(f'Weighted F1      : {result["weighted_f1"]:.4f}%'); print(f'Best val accuracy: {result["best_val_accuracy"]:.4f}%'); print(f'Best epoch       : {result["best_epoch"]}'); print(f'Seed time        : {result["seed_seconds"]/60:.2f} min')
    shutil.rmtree(tmp,ignore_errors=True)
    del tr,va,te,tr_ds,va_ds,te_ds,m,opt,sched,scaler; gc.collect(); torch.cuda.empty_cache()
    return result

def save_summary(outroot,seeds):
    rows=[]
    for s in seeds:
        p=outroot/f'seed_{s}'/'final_metrics.json'
        if p.exists(): rows.append(json.loads(p.read_text()))
    if not rows: return
    df=pd.DataFrame(rows).sort_values('split_seed'); df.to_csv(outroot/'all_split_results.csv',index=False)
    print('\n'+'='*100); print('COMPLETED SPLIT RESULTS'); print('='*100)
    cols=['split_seed','accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','best_val_accuracy','best_epoch','seed_seconds']
    print(df[cols].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    if len(df)==len(seeds):
        out=[]; print('\nMEAN ± SD')
        for met in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']:
            vals=df[met].to_numpy(float); mean=vals.mean(); sd=vals.std(ddof=1); half=2.7764451051977987*sd/math.sqrt(len(vals))
            print(f'{met:20s}: {mean:.4f} ± {sd:.4f}% [95% CI {mean-half:.4f}, {mean+half:.4f}]')
            out.append({'metric':met,'mean':mean,'sd':sd,'ci95_low':mean-half,'ci95_high':mean+half})
        pd.DataFrame(out).to_csv(outroot/'summary_mean_sd_ci.csv',index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='trashnet')
    ap.add_argument('--output',default=str(Path.home()/'trashnet_eva02_stratified_original_results'))
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--eval-workers',type=int,default=4)
    ap.add_argument('--prefetch',type=int,default=4)
    ap.add_argument('--seeds',nargs='+',type=int,default=SPLIT_SEEDS,choices=[0,1,2,3,4])
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU not detected.')
    data=resolve_data_root(a.data); out=Path(a.output).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda:0')
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    set_seed(TRAINING_SEED)
    print('='*100); print('EVA02-BASE 448 | TRASHNET STRATIFIED 70/13/17 | HPC'); print('='*100)
    print(f'Data root           : {data}'); print(f'Output              : {out}'); print(f'PyTorch             : {torch.__version__}'); print(f'timm                : {timm.__version__}'); print(f'GPU                 : {torch.cuda.get_device_name(0)}'); print(f'GPU memory          : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB'); print(f'os.cpu_count()      : {os.cpu_count()}'); print(f'Training workers    : {a.workers}'); print(f'Eval workers        : {a.eval_workers}'); print(f'Prefetch factor     : {a.prefetch}'); print(f'Batch               : {BATCH_SIZE}'); print(f'Accumulation        : {ACCUMULATION_STEPS}'); print(f'Effective batch     : {BATCH_SIZE*ACCUMULATION_STEPS}'); print('Precision           : FP16'); print(f'Channels-last       : {USE_CHANNELS_LAST}'); print(f'Grad checkpointing  : {USE_GRAD_CHECKPOINTING}'); print(f'Split seeds         : {a.seeds}'); print('Large checkpoints   : persistent best_model.pt plus temporary /tmp copy')
    df,classes=discover_dataset(data); print('\nDataset class counts:'); print(df['class'].value_counts().reindex(classes).to_string()); print(f'\nTotal images: {len(df)} | Classes: {len(classes)}')
    decoded=cache_compressed_bytes(df); train_t,eval_t=build_transforms(); weights=get_pretrained_path()
    for s in a.seeds:
        run_seed(s,df,decoded,classes,train_t,eval_t,weights,out,device,a.workers,a.eval_workers,a.prefetch); save_summary(out,a.seeds)
    save_summary(out,SPLIT_SEEDS)
    print(f'\nAll requested seeds finished. Results: {out}')

if __name__=='__main__':
    main()
