"""Training/validation RGB cache, fixed 2k labels and exact clean clip IDs."""
import sys,json,time,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np,onnxruntime as ort
from video_data import index_zip,decode
sys.path.insert(0,'src')
FEATURE_VERSION='appearance-mnv2-3view-v1'
root=Path('work/appearance-views-data');root.mkdir(exist_ok=True);encoder=Path('work/appearance/encoder.qat_views.onnx')
while not Path('work/appearance/encoder-views.json').exists():time.sleep(2)
encoder_hash=hashlib.file_digest(encoder.open('rb'),'sha256').hexdigest()
spatial=root/'spatial';spatial.mkdir(exist_ok=True)
metadata=root/'encoder.json'
if metadata.exists() and json.loads(metadata.read_text())['sha256']!=encoder_hash:raise ValueError('Existing cache uses a different encoder')
metadata.write_text(Path('work/appearance/encoder-views.json').read_text())
manifest_bytes=Path('work/3d-data-v3/manifest.json').read_bytes();rows=[r for r in json.loads(manifest_bytes) if r['split'] in ('train','val')]
v=index_zip('work/wlasl_video_format.zip');p=index_zip('work/wlasl_pose_format.zip');assert all(r['clip_id'] in v and r['clip_id'] in p for r in rows)
o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;s=ort.InferenceSession(str(encoder),o,providers=['CPUExecutionProvider']);start=time.monotonic();new=[]
# Bound submitted work so decoded frames never accumulate across the corpus.
with ThreadPoolExecutor(max_workers=3) as pool:
 pending={};submitted=0
 for i,row in enumerate(rows):
  while submitted<min(len(rows),i+6):
   r=rows[submitted]
   if not (root/(r['clip_id']+'.record.json')).exists():pending[submitted]=pool.submit(decode,r,v,p,16)
   submitted+=1
  target=root/(row['clip_id']+'.npy');record=root/(row['clip_id']+'.record.json')
  if record.exists():
   if not target.exists() or not (spatial/target.name).exists():raise ValueError('Missing tensor for completed record')
  else:
   b,h,m,info=pending.pop(i).result();body,bodymap=s.run(None,{'images':b});hand,handmap=s.run(None,{'images':h.reshape(-1,3,96,96)});hand=hand.reshape(16,2,1280)*m[:,:,None]
   def quadrants(x):
    height,width=x.shape[-2:];ys=[slice(0,(height+1)//2),slice(height//2,height)];xs=[slice(0,(width+1)//2),slice(width//2,width)]
    return np.stack([x[:,:,yy,xx].mean((2,3)) for yy in ys for xx in xs],axis=1).reshape(len(x),-1)
   bodyspatial=quadrants(bodymap);handspatial=quadrants(handmap).reshape(16,2,5120)*m[:,:,None]
   detail=np.concatenate((bodyspatial,handspatial[:,0],handspatial[:,1],m),axis=1).T.astype(np.float16)
   assert detail.shape==(15362,16) and np.isfinite(detail).all()
   starget=spatial/target.name;temporary=starget.with_suffix('.tmp.npy');np.save(temporary,detail);os.replace(temporary,starget)
   features=np.concatenate((body,hand[:,0],hand[:,1],m),axis=1).T.astype(np.float16)
   if features.shape!=(3842,16) or not np.isfinite(features).all():raise ValueError('Invalid embeddings')
   temporary=target.with_suffix('.tmp.npy');np.save(temporary,features);os.replace(temporary,target)
   record.write_text(json.dumps(dict(clip_id=row['clip_id'],split=row['split'],**info)))
  new.append(dict(row,features=target.name))
  if (i+1)%200==0:
   elapsed=time.monotonic()-start;print(json.dumps({'processed':i+1,'total':len(rows),'seconds':elapsed,'estimated_seconds_left':elapsed/(i+1)*(len(rows)-i-1)}),flush=True)
(root/'manifest.json').write_text(json.dumps(new,indent=2));(root/'labels.json').write_bytes(Path('work/3d-data-v3/labels.json').read_bytes())
reports=[json.loads((root/(r['clip_id']+'.record.json')).read_text()) for r in rows]
coverage=dict(feature_version=FEATURE_VERSION,frames=16,input_features=3842,classes=2000,counts=dict(Counter(r['split'] for r in rows)),encoder_sha256=encoder_hash,source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),hand_aligned_clips=sum(r['hand_frame_alignment'] for r in reports),unaligned_policy='Retain every clip, full-frame appearance only, zero masked hand features',test_payloads_read=False)
(root/'coverage.json').write_text(json.dumps(coverage,indent=2))
(spatial/'manifest.json').write_text(json.dumps(new,indent=2));(spatial/'labels.json').write_bytes((root/'labels.json').read_bytes());(spatial/'encoder.json').write_bytes(metadata.read_bytes());(spatial/'coverage.json').write_text(json.dumps(dict(coverage,feature_version='appearance-mnv2-quad-v1',input_features=15362),indent=2));print(json.dumps(coverage),flush=True)
