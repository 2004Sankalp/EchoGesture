"""Frozen public-pose evaluation; video/device metrics require separate runs."""
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np

from echogesture.engine import EnsembleRecognizer
from echogesture.geometry import stretch_vertical
parser=argparse.ArgumentParser();parser.add_argument('--v2-data',required=True);parser.add_argument('--v3-data',required=True);parser.add_argument('--split',choices=['val','test'],required=True);parser.add_argument('--output',required=True);parser.add_argument('--ensemble-directory',help='Frozen research ensemble; default is packaged ensemble');parser.add_argument('--manifest',help='Audited frozen split manifest');args=parser.parse_args()
engine=EnsembleRecognizer(directory=args.ensemble_directory);split=args.split;rows=None;combined=None;times=[]
for member,session in zip(engine.members,engine.sessions):
 root=Path(args.v3_data if member['feature_version'].endswith('v3') else args.v2_data)
 current=[r for r in json.loads((Path(args.manifest) if args.manifest else root/'manifest.json').read_text()) if r['split']==split]
 if rows is None:rows=current
 assert [r['clip_id'] for r in rows]==[r['clip_id'] for r in current]
 logits=[]
 for row in current:
  x=np.load(root/row['features'],allow_pickle=False).astype(np.float32);start=time.perf_counter_ns()
  y=sum(session.run(None,{'features3d':(x if s==1 else stretch_vertical(x,s))[None]})[0][0] for s in member['scales'])/len(member['scales']);logits.append(y)
  times.append((time.perf_counter_ns()-start)/1e6)
 contribution=member['weight']*np.stack(logits);combined=contribution if combined is None else combined+contribution
truth=np.array([r['class_id'] for r in rows]);top=np.argsort(-combined,axis=1)[:,:5];correct=top[:,0]==truth
predictions=[{'clip_id':r['clip_id'],'truth':r['label'],'prediction':engine.labels[int(t[0])],'top5':[engine.labels[int(i)] for i in t]} for r,t in zip(rows,top)]
report={'scope':'Frozen INT8-QDQ ensemble, full 2000-class public inferred-XYZ poses; not raw-video or organiser test. Exact duplicate records removed across train/val/test; signers overlap.','split':split,'clips':len(rows),'classes':2000,'top1_accuracy':100*float(correct.mean()),'top5_accuracy':100*float(np.any(top==truth[:,None],axis=1).mean()),'macro_accuracy':100*float(np.mean([correct[truth==c].mean() for c in np.unique(truth)])),'classes_with_eval_clips':len(np.unique(truth)),'model_asset_bytes':engine.model_asset_bytes,'ensemble':engine.config,'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in engine.model_paths},'predictions':predictions}
out=Path(args.output);out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','ensemble','artifacts']},indent=2))
