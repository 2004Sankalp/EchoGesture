"""Graph inputs from the existing, portable 3D feature ABI; no face required.

Graph channels are normalized XY and binary observed availability. Actual
inferred XYZ and 3D finger geometry also reach the separate geometry head.
Unavailable ears stay masked; they are never synthesized.
"""
import numpy as np

def from_features(features,frames=32):
 f=np.asarray(features,np.float32);assert f.shape==(308,48)
 indices=np.rint(np.linspace(0,47,frames)).astype(int);g=f[:,indices]
 body=np.zeros((frames,9,3),np.float32);q=g[:21].reshape(7,3,frames).transpose(2,0,1);present=g[21:28].T>.5;shoulders=present[:,1]&present[:,2]
 for source,target in enumerate([0,3,4,5,6,7,8]):body[:,target,:2]=q[:,source,:2];body[:,target,2]=present[:,source]*shoulders
 coords=body[body[:,:,2]>.3,:2];scale=0.
 if len(coords)>=4:
  low=coords.min(0);high=coords.max(0);scale=float(max(high-low))
 if scale>1e-6:
  origin=(low+high-scale)/2;body[:,:,:2]=((body[:,:,:2]-origin)/scale-.5)*2
 else:body[:]=0
 body[body[:,:,2]<=.3]=0;result={'body':np.clip(body,-1,1)[None]}
 for name,start in [('left',180),('right',244)]:
  raw=g[start:start+63].reshape(21,3,frames).transpose(2,0,1);valid=np.any(raw!=0,axis=(1,2))&shoulders
  hand=np.zeros((frames,21,3),np.float32)
  if scale>1e-6:
   hand[:,:,:2]=(raw[:,:,:2]-raw[:,:1,:2])/scale;hand[:,:,2]=valid[:,None];hand[~valid]=0
  result[name]=np.clip(hand,-1,1)[None]
 return result
