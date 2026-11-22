"""Local RGB preprocessing, shared by offline training and eventual runtime.

Normalized hand XY plus visibility are sufficient; missing hands stay masked.
No images are uploaded or saved by these functions.
"""
import cv2
import numpy as np
MEAN=np.array([.485,.456,.406],np.float32)
STD=np.array([.229,.224,.225],np.float32)
FEATURE_VERSION='appearance-mnv3-3view-v1'

def normalized(image):return ((image/255.-MEAN)/STD).transpose(2,0,1).astype(np.float32)

def views(bgr,hands=None,scores=None):
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
 if h<1 or w<1:raise ValueError('Empty image')
 scale=160/max(h,w);rw=max(1,round(w*scale));rh=max(1,round(h*scale))
 body=np.full((160,160,3),MEAN*255,dtype=np.float32)
 x=(160-rw)//2;y=(160-rh)//2
 body[y:y+rh,x:x+rw]=cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_LINEAR)
 crops=[];valid=[]
 for side in range(2):
  result=np.zeros((3,96,96),np.float32);present=False
  if hands is not None:
   points=np.asarray(hands[side],np.float32)
   confidence=np.asarray(scores[side],np.float32) if scores is not None else np.ones(21,np.float32)
   mask=np.isfinite(points).all(-1)&np.isfinite(confidence)&(confidence>=.2)&(points>-.2).all(-1)&(points<1.2).all(-1)
   if mask.sum()>=6 and mask[0]:
    p=points[mask]*[w,h];lo=p.min(0);hi=p.max(0);centre=(lo+hi)*.5;extent=float(max(hi-lo)*1.6)
    if 8<=extent<=2*max(h,w):
     factor=96/extent;matrix=np.array([[factor,0,48-factor*centre[0]],[0,factor,48-factor*centre[1]]],np.float32)
     crop=cv2.warpAffine(rgb.astype(np.float32),matrix,(96,96),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=tuple(float(v) for v in MEAN*255))
     result=normalized(crop);present=True
  crops.append(result);valid.append(float(present))
 return normalized(body),np.stack(crops),np.array(valid,np.float32)
