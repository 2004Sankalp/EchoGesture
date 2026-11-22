"""Independent RTMPose whole-body research features; no inferred 3D claims.

Input is normalized image XY, COCO WholeBody 133 ordering. This expert must
be fused by clip ID, never concatenated frame-by-frame with another extractor.
"""
import numpy as np
FEATURE_VERSION='rtmpose-local2d-v1'
FEATURE_DIM=385

def prepare2d(points,scores,frames=48):
 p=np.asarray(points,dtype=np.float32).reshape(-1,133,2).copy()
 s=np.asarray(scores,dtype=np.float32).reshape(-1,133)
 if len(p)==0 or len(p)!=len(s):raise ValueError('Empty/mismatched sequence')
 valid=np.isfinite(p).all(-1)&np.isfinite(s)&(s>=.2)
 p=np.nan_to_num(p,nan=0.,posinf=0.,neginf=0.)
 def canonical(q,m,origin,axis,scale):
  ex=axis/np.maximum(np.linalg.norm(axis,axis=-1,keepdims=True),1e-4)
  ey=np.stack((-ex[:,1],ex[:,0]),-1)
  local=q-origin[:,None]
  result=np.stack(((local*ex[:,None]).sum(-1),(local*ey[:,None]).sum(-1)),-1)/np.maximum(scale[:,None,None],1e-4)
  return np.clip(result,-8,8)*m[:,:,None]
 shoulder=p[:,6]-p[:,5];scale=np.linalg.norm(shoulder,axis=-1)
 bodyok=valid[:,5]&valid[:,6]&(scale>.01)
 bodymask=valid[:,:17]&bodyok[:,None]
 body=canonical(p[:,:17],bodymask,(p[:,5]+p[:,6])/2,shoulder,scale)
 blocks=[body.reshape(len(p),-1)];masks=[bodymask]
 for start in (91,112):
  q=p[:,start:start+21];m=valid[:,start:start+21]
  axis=q[:,9]-q[:,0]
  hs=(np.linalg.norm(q[:,5]-q[:,17],axis=-1)+np.linalg.norm(axis,axis=-1))*.5
  ok=m[:,0]&m[:,5]&m[:,9]&m[:,17]&(hs>.003)
  m=m&ok[:,None]
  blocks.append(canonical(q,m,q[:,0],axis,hs).reshape(len(p),-1));masks.append(m)
 face=p[:,23:91];fm=valid[:,23:91]
 axis=face[:,45]-face[:,36];fs=np.linalg.norm(axis,axis=-1)
 ok=fm[:,36]&fm[:,45]&(fs>.005)
 fm=fm&ok[:,None]
 blocks.append(canonical(face,fm,(face[:,36]+face[:,45])*.5,axis,fs).reshape(len(p),-1));masks.append(fm)
 # Wrist position relative to the torso retains hand location lost by local shape.
 wrists=p[:,[91,112]];wm=valid[:,[91,112]]&bodyok[:,None]
 blocks.append(canonical(wrists,wm,(p[:,5]+p[:,6])*.5,shoulder,scale).reshape(len(p),-1))
 blocks.extend(np.asarray(m,dtype=np.float32) for m in masks)
 f=np.concatenate(blocks,axis=1)
 assert f.shape[1]==FEATURE_DIM
 t=np.linspace(0,len(f)-1,frames);lo=t.astype(int);hi=np.minimum(lo+1,len(f)-1);w=(t-lo)[:,None]
 return (f[lo]*(1-w)+f[hi]*w).T.astype(np.float32)


def prepare_transfer2d(points,scores,frames=48):
 """512-channel 2D transfer profile. Zero Z is missing depth, not measured 3D.

 The first 308 channels reuse the existing geometry ABI to initialize a sign
 encoder. The appended 204 channels contain face XY and its visibility mask.
 This profile cannot replace the required real inferred-XYZ hand stream.
 """
 from .features3d import prepare3d
 p=np.asarray(points,dtype=np.float32).reshape(-1,133,2)
 s=np.asarray(scores,dtype=np.float32).reshape(-1,133)
 valid=np.isfinite(p).all(-1)&np.isfinite(s)&(s>=.2)
 p=np.nan_to_num(p,nan=0.,posinf=0.,neginf=0.)
 q=np.zeros((len(p),75,3),dtype=np.float32)
 for target,source in zip([0,11,12,13,14,15,16],[0,5,6,7,8,9,10]):
  q[:,target,:2]=p[:,source]*valid[:,source,None]
 for target,source in [(33,91),(54,112)]:
  m=valid[:,source:source+21]
  complete=m[:,[0,5,9,17]].all(1)
  q[:,target:target+21,:2]=p[:,source:source+21]*m[:,:,None]*complete[:,None,None]
 base=prepare3d(q,frames)
 local=prepare2d(points,scores,frames)
 return np.concatenate((base,local[118:254],local[317:385]),axis=0)
