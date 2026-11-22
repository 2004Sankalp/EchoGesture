"""CPU training. Checkpoint selection uses validation only; never test accuracy."""
import argparse
import json
import hashlib
import random
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from echogesture.temporal3d import Temporal3D
from echogesture.features3d import FEATURE_VERSION


def save_checkpoint(state,path):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def score(model, loader, device):
    total=correct=0; losses=0.
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            logits=model(x.to(device,dtype=torch.float32)); y=y.to(device)
            total+=len(y);correct+=(logits.argmax(1)==y).sum().item()
            losses+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
    return dict(top1_accuracy=100*correct/total,loss=losses/total,clips=total)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',help='Frozen manifest snapshot within the feature cache')
    parser.add_argument('--coverage',help='Matching frozen dataset provenance snapshot')
    parser.add_argument('--tensor-cache',help='Shared float16 arrays made by cache_training_tensors.py')
    parser.add_argument('--epochs',type=int,default=120);parser.add_argument('--patience',type=int,default=20)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--channels',type=int,default=128);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume');parser.add_argument('--max-seconds',type=int,default=7200)
    parser.add_argument('--schedule',choices=['plateau','cosine'],default='plateau')
    parser.add_argument('--classifier',choices=['linear','cosine'],default='linear')
    parser.add_argument('--architecture',choices=['mixer','attention','anatomy','anatomy_context','anatomy_kinematic','motion','appearance'],default='mixer')
    parser.add_argument('--lr',type=float,default=0.0015)
    parser.add_argument('--initialize',help='Compatible clean-training weights; optimizer and schedule start fresh')
    parser.add_argument('--allow-feature-transfer',action='store_true',help='Explicitly fine-tune matching-width weights for a new coordinate version')
    parser.add_argument('--aspect-augment',action='store_true',help='Recompute 3D geometry after sampled vertical coordinate stretch')
    parser.add_argument('--balanced',action='store_true',help='Training-only inverse-frequency sampling; validation is unchanged')
    parser.add_argument('--teacher',help='Train-only ID-aligned teacher NPZ; held-out IDs rejected')
    parser.add_argument('--distillation-weight',type=float,default=.4)
    parser.add_argument('--distillation-temperature',type=float,default=3.)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(args.manifest) if args.manifest else root/'manifest.json'
    manifest_bytes=manifest_path.read_bytes()
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    manifest=json.loads(manifest_bytes);labels=json.loads((root/'labels.json').read_text())
    datasets={}
    cache=Path(args.tensor_cache) if args.tensor_cache else None
    if cache and json.loads((cache/'cache.json').read_text())['manifest_sha256']!=manifest_hash:
        raise ValueError('Tensor cache belongs to a different manifest')
    # Test tensors are intentionally never loaded by the training script.
    for split in ('train','val'):
        rows=[r for r in manifest if r['split']==split]
        if not rows:raise ValueError(f'No {split} clips')
        # Keep the exact stored float16 values in RAM; cast each batch before
        # arithmetic. This avoids doubling large caches on CPU-only laptops.
        y=np.array([r['class_id'] for r in rows],dtype=np.int64)
        if cache:
            # Copy-on-write mapping is writable to PyTorch but is never mutated:
            # DataLoader stacks rows into a separate batch before float casting.
            x=np.load(cache/(split+'.npy'),mmap_mode='c',allow_pickle=False)
            cached_y=np.load(cache/(split+'_labels.npy'),allow_pickle=False)
            if len(x)!=len(rows) or not np.array_equal(cached_y,y):
                raise ValueError('Cached clip/label alignment differs')
        else:x=np.stack([np.load(root/r['features'],allow_pickle=False) for r in rows])
        datasets[split]=TensorDataset(torch.from_numpy(x),torch.from_numpy(y))
    if len({r['class_id'] for r in manifest if r['split']=='train'})!=len(labels):
        raise ValueError('Every declared class must have actual training clips')
    teacher_hash=None
    if args.teacher:
        with open(args.teacher,'rb') as teacher_file:teacher_hash=hashlib.file_digest(teacher_file,'sha256').hexdigest()
        if not 0<=args.distillation_weight<=1 or args.distillation_temperature<=0:raise ValueError('Invalid distillation parameters')
        from echogesture.distillation import load_training_teacher
        teacher,teacher_valid=load_training_teacher(args.teacher,[r for r in manifest if r['split']=='train'],labels)
        datasets['train']=TensorDataset(*datasets['train'].tensors,torch.from_numpy(teacher),torch.from_numpy(teacher_valid))
    loaders={s:DataLoader(d,batch_size=args.batch,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
    if args.balanced:
        truth=datasets['train'].tensors[1]
        weights=1/torch.bincount(truth,minlength=len(labels)).double()[truth]
        loaders['train']=DataLoader(datasets['train'],batch_size=args.batch,
            sampler=WeightedRandomSampler(weights,len(truth),replacement=True),num_workers=0)
    coverage=json.loads((Path(args.coverage) if args.coverage else root/'coverage.json').read_text())
    input_features=datasets['train'].tensors[0].shape[1]
    model=Temporal3D(len(labels),args.channels,datasets['train'].tensors[0].shape[-1],input_features,args.classifier,args.architecture)
    initial_hash=None
    if args.initialize and not args.resume:
        initial=torch.load(args.initialize,map_location='cpu',weights_only=True)
        snapshot={key:initial[key] for key in ('model','labels','feature_version','channels','frames','best','best_epoch') if key in initial}
        save_checkpoint(snapshot,out/'initial_weights.pt')
        initial_hash=hashlib.sha256((out/'initial_weights.pt').read_bytes()).hexdigest()
        if initial['labels']!=labels or (initial['feature_version']!=coverage['feature_version'] and not args.allow_feature_transfer):
            raise ValueError('Initial checkpoint vocabulary/features differ')
        initial_state=dict(initial['model'])
        if args.classifier=='cosine':initial_state.pop('head.2.bias',None)
        model.load_state_dict(initial_state,strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=4,min_lr=0.00001)
        if args.schedule=='plateau' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=0.00002))
    first=0;best=-1.;best_epoch=-1;history=[]
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state.get('teacher_sha256')!=teacher_hash:raise ValueError('Resume teacher changed')
        if state['labels']!=labels:raise ValueError('Resume vocabulary mismatch')
        if state.get('dataset_manifest_sha256',manifest_hash)!=manifest_hash:
            raise ValueError('Resume dataset changed; explicitly fine-tune with --initialize instead')
        if state['feature_version']!=coverage['feature_version'] or state.get('classifier','linear')!=args.classifier or state.get('architecture','mixer')!=args.architecture:
            raise ValueError('Resume features/classifier/architecture mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        first=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch'];history=state['history']
        if 'rng_torch' in state:torch.set_rng_state(state['rng_torch'])
        initial_hash=state.get('initialization_sha256')
    start=time.monotonic()
    for epoch in range(first,args.epochs):
        tick=time.monotonic();model.train();loss_total=0.;correct=total=0
        for batch in loaders['train']:
            x,y=batch[:2]
            x=x.float()
            if args.aspect_augment:
                from echogesture.augment3d import stretch_vertical
                factors=0.65+torch.rand(len(x))*0.95
                factors=torch.where(torch.rand(len(x))<0.5,factors,torch.ones_like(factors))
                x=stretch_vertical(x,factors)
            # Small noise only on observed channels. No label-changing hand swap.
            x=x+torch.randn_like(x)*0.02*(x!=0)
            # Clip-level feature dropout limits reliance on one noisy joint.
            x=x*(torch.rand(x.shape[0],x.shape[1],1)>0.05)
            if torch.rand(())<0.35:
                start_frame=int(torch.randint(0,5,()).item());end_frame=x.shape[-1]-int(torch.randint(0,5,()).item())
                x=nn.functional.interpolate(x[:,:,start_frame:end_frame],size=x.shape[-1],mode='linear',align_corners=False)
            logits=model(x);ce=nn.functional.cross_entropy(logits,y,label_smoothing=0.1,reduction='none')
            if args.teacher:
                t=args.distillation_temperature
                target=nn.functional.softmax(batch[2].float()/t,dim=1)
                kl=nn.functional.kl_div(nn.functional.log_softmax(logits/t,dim=1),target,reduction='none').sum(1)*(t*t)
                amount=args.distillation_weight*batch[3]
                loss=((1-amount)*ce+amount*kl).mean()
            else:loss=ce.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_total+=loss.item()*len(y)
        validation=score(model,loaders['val'],'cpu')
        if args.schedule=='plateau':scheduler.step(validation['top1_accuracy'])
        else:scheduler.step()
        entry=dict(epoch=epoch+1,train_top1=100*correct/total,train_loss=loss_total/total,
                   val_top1=validation['top1_accuracy'],val_loss=validation['loss'],seconds=time.monotonic()-tick)
        history.append(entry)
        improved=validation['top1_accuracy']>best
        if improved:best=validation['top1_accuracy'];best_epoch=epoch
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                   best=best,best_epoch=best_epoch,labels=labels,feature_version=FEATURE_VERSION,channels=args.channels,
                   frames=datasets['train'].tensors[0].shape[-1],history=history,seed=args.seed,selection='validation top1 only',
                   input_features=input_features,run_arguments=vars(args),rng_torch=torch.get_rng_state())
        state['dataset_manifest_sha256']=manifest_hash
        state['teacher_sha256']=teacher_hash
        state['classifier']=args.classifier
        state['architecture']=args.architecture
        if initial_hash:state['initialization_sha256']=initial_hash
        state['feature_version']=coverage['feature_version']
        save_checkpoint(state,out/'last.pt')
        if improved:save_checkpoint(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(entry),flush=True)
        if epoch-best_epoch>=args.patience or time.monotonic()-start>args.max_seconds:break
    print(json.dumps(dict(best_validation_top1=best,best_epoch=best_epoch+1,parameters=sum(p.numel() for p in model.parameters()),test_used=False)),flush=True)


if __name__=='__main__':main()
