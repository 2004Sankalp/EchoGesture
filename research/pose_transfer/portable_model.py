"""Single portable 3D input, graph transfer features and word predictions."""
import torch
from torch import nn
from encoder import PoseEncoder
from mp_preprocessor import MPGraphInputs
from head import GeometryPoseHead
from echogesture.temporal3d import finger_kinematics

def sequence_statistics(x):
 mean=x.mean(2)
 return torch.cat((mean,x.amax(2),torch.sqrt(((x-mean[:,:,None])**2).mean(2)+1e-6),(x[:,:,1:]-x[:,:,:-1]).abs().mean(2)),1)

def geometry_statistics(x):
 mean=x.mean(2);kin=finger_kinematics(x);km=kin.mean(2)
 return torch.cat((mean,torch.sqrt(((x-mean[:,:,None])**2).mean(2)+1e-6),x.abs().amax(2),km,torch.sqrt(((kin-km[:,:,None])**2).mean(2)+1e-6)),1)

class PortablePoseModel(nn.Module):
 def __init__(self,encoder_state,head_state):
  super().__init__();self.inputs=MPGraphInputs(32);self.encoder=PoseEncoder(encoder_state,include_face=False);self.head=GeometryPoseHead(3072,head_state['channels']);self.head.load_state_dict(head_state['model'])
 def forward(self,features):
  body,left,right=self.inputs(features);sequence,_=self.encoder(body,left,right)
  stats=torch.cat((sequence_statistics(sequence),geometry_statistics(features)),1)
  return self.head(stats)
