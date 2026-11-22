import sys
import numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from echogesture.temporal3d import finger_kinematics

def sequence_summary(x):
 mean=x.mean(1);std=np.sqrt(((x-mean[:,None])**2).mean(1)+1e-6)
 velocity=np.abs(np.diff(x,axis=1)).mean(1) if x.shape[1]>1 else np.zeros(len(x),np.float32)
 return np.concatenate((mean,x.max(1),std,velocity)).astype(np.float32)

def geometry_summary(x):
 x=np.asarray(x,np.float32);mean=x.mean(1);std=np.sqrt(((x-mean[:,None])**2).mean(1)+1e-6)
 kin=finger_kinematics(torch.from_numpy(x[None])).numpy()[0];km=kin.mean(1);ks=np.sqrt(((kin-km[:,None])**2).mean(1)+1e-6)
 return np.concatenate((mean,std,np.abs(x).max(1),km,ks)).astype(np.float32)
