"""Same portable palm3d feature ABI for training, desktop and Android."""
import json,time,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,onnxruntime as ort
from mp_parts import from_features
from feature_statistics import sequence_summary,geometry_summary
root=Path('work/unisign-study/features-mp');root.mkdir(exist_ok=True);data=Path('work/3d-data-citizen-v3');torch.set_num_threads(1)
while not Path('work/unisign-study/encoder-mp2-quantization.json').exists():time.sleep(2)
rows=[r for r in json.load(open(data/'manifest-video-clean.json')) if r['split'] in ('train','val')];lookup={};maps={}
for split in ['train','val']:
 split_rows=[r for r in rows if r['split']==split];maps[split]=np.load(data/('shared-video-clean/'+split+'.npy'),mmap_mode='r',allow_pickle=False)
 for i,r in enumerate(split_rows):lookup[r['clip_id']]=(split,i)
encoder=Path('work/unisign-study/encoder-mp2.int8.onnx');fingerprint=hashlib.file_digest(encoder.open('rb'),'sha256').hexdigest();plan=dict(encoder_sha256=fingerprint,manifest_sha256=hashlib.sha256((data/'manifest-video-clean.json').read_bytes()).hexdigest(),feature_version='OpenASL-MP-palm3d32-stat-v1',frames=32)
if (root/'plan.json').exists() and json.load(open(root/'plan.json'))!=plan:raise ValueError('Cache provenance differs')
(root/'plan.json').write_text(json.dumps(plan,indent=2));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;session=ort.InferenceSession(str(encoder),options,providers=['CPUExecutionProvider']);started=time.monotonic()
def prepare(row):
 split,index=lookup[row['clip_id']];x=np.array(maps[split][index],dtype=np.float32);return from_features(x),geometry_summary(x)
with ThreadPoolExecutor(max_workers=2) as pool:
 pending={};submitted=0
 for i,row in enumerate(rows):
  while submitted<min(len(rows),i+4):
   r=rows[submitted]
   if not (root/(r['clip_id']+'.record.json')).exists():pending[submitted]=pool.submit(prepare,r)
   submitted+=1
  record=root/(row['clip_id']+'.record.json')
  if not record.exists():
   feed,geometry=pending.pop(i).result();raw,tail=session.run(None,feed);value=np.concatenate((sequence_summary(raw[0]),geometry)).astype(np.float16)
   assert value.shape==(4088,) and np.isfinite(value).all()
   for suffix,x in [('noface',value),('tail',tail[0].astype(np.float16))]:
    target=root/(row['clip_id']+'.'+suffix+'.npy');temporary=target.with_suffix('.tmp.npy');np.save(temporary,x);os.replace(temporary,target)
   temporary=record.with_suffix('.tmp.json');temporary.write_text(json.dumps(dict(clip_id=row['clip_id'],split=row['split'],valid_frames=32)));os.replace(temporary,record)
  if (i+1)%200==0:
   elapsed=time.monotonic()-started;print(json.dumps({'processed':i+1,'total':len(rows),'seconds':elapsed,'estimated_seconds_left':elapsed/(i+1)*(len(rows)-i-1)}),flush=True)
(root/'manifest.json').write_text(json.dumps(rows,indent=2));(root/'coverage.json').write_text(json.dumps(dict(plan,train=30730,val=3865,classes=2000,test_not_read=True,geometry_features=1016,pose_features=3072,input_profile='Canonical3D ABI -> body/hands XY and binary availability; real XYZ/finger geometry also in head'),indent=2));print('Completed portable MP-3D transfer feature cache',flush=True)
