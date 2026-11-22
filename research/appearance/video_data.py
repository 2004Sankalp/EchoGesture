"""Decode exact clean training/validation IDs; never load test video payloads."""
import sys,io,zipfile,tempfile,os,threading
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,'src')
from prepare_public_poses import ArrayUnpickler
from echogesture.appearance import views

def index_zip(path):
 with zipfile.ZipFile(path) as z:return {Path(n).stem.split('-wlasl-',1)[0]:n for n in z.namelist() if Path(n).suffix.lower() in ('.mp4','.pkl')}

_local=threading.local()

def archives():
 if not hasattr(_local,'videos'):
  _local.videos=zipfile.ZipFile('work/wlasl_video_format.zip')
  _local.poses=zipfile.ZipFile('work/wlasl_pose_format.zip')
 return _local.videos,_local.poses

def decode(row,video_names,pose_names,frames=16):
 if row['split'] not in ('train','val'):raise ValueError('Held-out test payload forbidden in research extraction')
 clip=row['clip_id']
 videos,poses=archives();raw=videos.read(video_names[clip])
 d=ArrayUnpickler(io.BytesIO(poses.read(pose_names[clip]))).load()
 points=np.asarray(d['keypoints'],np.float32).reshape(-1,133,2);scores=np.asarray(d['scores'],np.float32).reshape(-1,133)
 handle=tempfile.NamedTemporaryFile(suffix='.mp4',dir='work/appearance',delete=False)
 try:
  handle.write(raw);handle.close();cap=cv2.VideoCapture(handle.name);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if count<=0:raise ValueError('Missing video frame count '+clip)
  indices=np.rint(np.linspace(0,count-1,frames)).astype(int);wanted=set(indices.tolist());decoded={};i=0
  while i<=indices[-1]:
   success,frame=cap.read()
   if not success:break
   if i in wanted:decoded[i]=frame
   i+=1
  cap.release()
  if not decoded:raise ValueError('No decoded frames '+clip)
  aligned=count==len(points) and i==count
  body=[];hands=[];masks=[]
  for index in indices:
   image=decoded.get(int(index))
   if image is None:image=decoded[max(decoded)]
   q=points[index,91:133].reshape(2,21,2) if aligned else None
   c=scores[index,91:133].reshape(2,21) if aligned else None
   b,h,m=views(image,q,c);body.append(b);hands.append(h);masks.append(m)
  return np.stack(body),np.stack(hands),np.stack(masks),dict(video_frames=count,decoded_frames=i,pose_frames=len(points),hand_frame_alignment=bool(aligned),valid_hand_fraction=float(np.mean(masks)))
 finally:
  handle.close();os.unlink(handle.name)
