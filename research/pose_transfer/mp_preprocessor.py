"""ONNX-portable form of the tested 308-channel geometry-to-graph adapter."""
import torch
from torch import nn
class MPGraphInputs(nn.Module):
 def __init__(self,frames=32):
  super().__init__();self.register_buffer('indices',torch.linspace(0,47,frames).round().long())
 def forward(self,features):
  x=features.index_select(2,self.indices);batch=x.shape[0];time=x.shape[-1]
  q=x[:,:21].reshape(batch,7,3,time).permute(0,3,1,2);present=x[:,21:28].transpose(1,2)>.5;shoulders=present[:,:,1]&present[:,:,2];score=present&shoulders[:,:,None]
  zeros=torch.zeros_like(q[:,:,0]);zero_score=torch.zeros_like(score[:,:,0]);body_xy=torch.stack((q[:,:,0,:2],zeros[:,:,:2],zeros[:,:,:2],q[:,:,1,:2],q[:,:,2,:2],q[:,:,3,:2],q[:,:,4,:2],q[:,:,5,:2],q[:,:,6,:2]),dim=2)
  body_score=torch.stack((score[:,:,0],zero_score,zero_score,score[:,:,1],score[:,:,2],score[:,:,3],score[:,:,4],score[:,:,5],score[:,:,6]),dim=2)
  low=torch.where(body_score[:,:,:,None],body_xy,torch.full_like(body_xy,10000.)).amin((1,2));high=torch.where(body_score[:,:,:,None],body_xy,torch.full_like(body_xy,-10000.)).amax((1,2));scale=(high-low).amax(1);valid=(body_score.sum((1,2))>=4)&(scale>1e-6);scale=torch.where(valid,scale,torch.ones_like(scale));origin=(low+high-scale[:,None])/2
  normalized=((body_xy-origin[:,None,None,:])/scale[:,None,None,None]-.5)*2
  body=torch.cat((normalized,body_score.float()[:,:,:,None]),dim=-1)
  body=torch.where((body_score&valid[:,None,None])[:,:,:,None],body,torch.zeros_like(body)).clamp(-1,1)
  hands=[]
  for start in [180,244]:
   raw=x[:,start:start+63].reshape(batch,21,3,time).permute(0,3,1,2);observed=(raw!=0).any((2,3))&shoulders&valid[:,None]
   xy=(raw[:,:,:,:2]-raw[:,:,:1,:2])/scale[:,None,None,None];confidence=observed.float()[:,:,None,None].expand(-1,-1,21,1);hand=torch.cat((xy,confidence),dim=-1);hand=torch.where(observed[:,:,None,None],hand,torch.zeros_like(hand));hands.append(hand.clamp(-1,1))
  return body,hands[0],hands[1]
