"""Cache only clean train/validation clips from the public whole-body archive."""
import argparse,json,zipfile,io,hashlib,os
from pathlib import Path
from collections import Counter
import numpy as np
from prepare_public_poses import ArrayUnpickler
from echogesture.features2d import prepare2d,prepare_transfer2d,FEATURE_VERSION

def main():
 a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--manifest',required=True);a.add_argument('--labels',required=True);a.add_argument('--output',required=True);a.add_argument('--transfer',action='store_true');args=a.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 version='rtmpose-xy-face-v1' if args.transfer else FEATURE_VERSION
 if (out/'coverage.json').exists() and json.loads((out/'coverage.json').read_text())['feature_version']!=version:
  raise ValueError('Use a fresh directory for a different feature profile')
 raw=Path(args.manifest).read_bytes();rows=[r for r in json.loads(raw) if r['split'] in ('train','val')]
 z=zipfile.ZipFile(args.archive);names={Path(n).stem:n for n in z.namelist() if n.endswith('.pkl')}
 empty=Counter();new=[]
 for i,row in enumerate(rows):
  target=out/(row['clip_id']+'.npy')
  if not target.exists():
   if row['clip_id'] not in names:raise ValueError('Missing clip '+row['clip_id'])
   d=ArrayUnpickler(io.BytesIO(z.read(names[row['clip_id']]))).load()
   f=(prepare_transfer2d if args.transfer else prepare2d)(d['keypoints'],d['scores'])
   temporary=target.with_suffix('.tmp.npy');np.save(temporary,f.astype(np.float16));os.replace(temporary,target)
  else:f=np.load(target,allow_pickle=False)
  if f.shape!=((512 if args.transfer else 385),48) or not np.isfinite(f).all():raise ValueError('Incompatible cached features: '+row['clip_id'])
  if not np.any(f):empty[row['split']]+=1
  new.append(dict(row,features=target.name))
  if (i+1)%1000==0:print(json.dumps({'processed':i+1,'total':len(rows)}),flush=True)
 (out/'manifest.json').write_text(json.dumps(new,indent=2));(out/'labels.json').write_bytes(Path(args.labels).read_bytes())
 coverage=dict(feature_version='rtmpose-xy-face-v1' if args.transfer else FEATURE_VERSION,frames=48,counts=dict(Counter(r['split'] for r in new)),classes=2000,source='Uni-Sign public RTMPose WholeBody XY; late-fusion only; no test tensors read',base_manifest_sha256=hashlib.sha256(raw).hexdigest(),empty_clips_retained=dict(empty))
 (out/'coverage.json').write_text(json.dumps(coverage,indent=2));print(json.dumps(coverage),flush=True)
if __name__=='__main__':main()
