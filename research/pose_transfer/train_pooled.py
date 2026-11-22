"""Wait for train/validation features, then train fresh 2000-class heads."""
import time,json,sys,random,hashlib,os,signal,argparse
from pathlib import Path
import numpy as np,torch
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from head import GeometryPoseHead
parser=argparse.ArgumentParser();parser.add_argument('--features',required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True);parser.add_argument('--mode',default='noface');parser.add_argument('--channels',type=int,default=256);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--initialize');parser.add_argument('--lr',type=float,default=.0015);parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--max-seconds',type=int,default=2400);args=parser.parse_args()
root=Path(args.features);outroot=Path(args.output);outroot.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(args.manifest).read_text());labels=json.loads((Path(__file__).resolve().parents[2]/'models/labels.json').read_text());torch.set_num_threads(1)
for mode,width,seed in [(args.mode,args.channels,args.seed)]:
 out=outroot;out.mkdir(exist_ok=True);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);loaders={};manifest={}
 for split in ['train','val']:
  r=[r for r in rows if r['split']==split];x=np.stack([np.load(root/(a['clip_id']+'.'+mode+'.npy'),allow_pickle=False) for a in r]);y=np.array([a['class_id'] for a in r],np.int64);dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(y));manifest[split]=r
  if split=='train':
   weights=1/torch.bincount(dataset.tensors[1],minlength=2000).double()[dataset.tensors[1]];loaders[split]=DataLoader(dataset,batch_size=128,sampler=WeightedRandomSampler(weights,len(y),replacement=True))
  else:loaders[split]=DataLoader(dataset,batch_size=256)
 pose_dim=len(loaders['train'].dataset.tensors[0][0])-1016;model=GeometryPoseHead(pose_dim,width).train()
 if args.initialize:
  initial=torch.load(args.initialize,map_location='cpu',weights_only=True)
  if initial['labels']!=labels or initial['pose_features']!=pose_dim or initial['channels']!=width:raise ValueError('Initialization differs')
  model.load_state_dict(initial['model'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.02);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4,min_lr=.00001);best=-1;best_epoch=0;history=[];start=time.monotonic()
 for epoch in range(args.epochs):
  tick=time.monotonic();model.train();correct=total=0;loss_sum=0.
  for x,y in loaders['train']:
   logits=model(x.float());loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.1);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=len(y);correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
  model.eval();pred=[];truth=[]
  with torch.inference_mode():
   for x,y in loaders['val']:pred.append(model(x.float()).numpy());truth.append(y.numpy())
  pred=np.concatenate(pred);truth=np.concatenate(truth);accuracy=100*float(np.mean(pred.argmax(1)==truth));scheduler.step(accuracy)
  entry=dict(epoch=epoch+1,train_top1=100*correct/total,train_loss=loss_sum/total,val_top1=accuracy,seconds=time.monotonic()-tick);history.append(entry)
  state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=max(best,accuracy),labels=labels,pose_features=pose_dim,channels=width,mode=mode,feature_version='OpenASL-INT8-clipstats+inferred-XYZ-geometry-v1',training_clips=len(manifest['train']),validation_clips=len(manifest['val']),encoder_sha256=json.load(open(root/('plan.json' if (root/'plan.json').exists() else 'coverage.json')))['encoder_sha256'],selection='Validation only, all2000 outputs; original WLASL test not read',history=history,manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),arguments=vars(args))
  temporary=out/'last.pt.tmp';torch.save(state,temporary);os.replace(temporary,out/'last.pt')
  if accuracy>best:
   best=accuracy;best_epoch=epoch+1;temporary=out/'best.pt.tmp';torch.save(state,temporary);os.replace(temporary,out/'best.pt');np.savez(out/'validation.npz',logits=pred.astype(np.float16),truth=truth,clip_ids=np.array([r['clip_id'] for r in manifest['val']]))
  (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(mode=mode,best=best,**entry)),flush=True)
  if epoch+1-best_epoch>=15 or time.monotonic()-start>args.max_seconds:break
 print(json.dumps({'mode':mode,'best_validation_top1':best,'best_epoch':best_epoch,'test_not_read':True}),flush=True)
