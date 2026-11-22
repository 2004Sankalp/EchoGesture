"""Own inference-only ST-GCN implementation for the licensed OpenASL weights.

Implements graph-convolution equations using checkpoint adjacency matrices;
no language decoder, tokenizer, publisher Python module or WLASL weights.
Architecture reference: Uni-Sign / CoSign. Inputs are XY + confidence, NOT XYZ.
"""
import torch
from torch import nn

def conv_from(state,prefix,padding=0):
 w=state[prefix+'.weight'].float();b=state.get(prefix+'.bias');layer=nn.Conv2d(w.shape[1],w.shape[0],w.shape[2:],padding=(padding,0),bias=b is not None)
 with torch.no_grad():
  layer.weight.copy_(w)
  if b is not None:layer.bias.copy_(b.float())
 return layer

def norm_from(state,prefix):
 layer=nn.BatchNorm2d(len(state[prefix+'.weight']),eps=1e-5)
 layer.load_state_dict({k:state[prefix+'.'+k].float() if k!='num_batches_tracked' else state[prefix+'.'+k] for k in layer.state_dict()})
 return layer

class GraphBlock(nn.Module):
 def __init__(self,state,prefix):
  super().__init__();self.graph=conv_from(state,prefix+'.gcn.conv');self.adjacency=nn.Parameter(state[prefix+'.gcn.A'].float());self.graph_norm=norm_from(state,prefix+'.gcn.bn');self.partitions=self.adjacency.shape[0]
  self.temporal=nn.Identity()
  if prefix+'.tcn.0.weight' in state:self.temporal=nn.Sequential(conv_from(state,prefix+'.tcn.0',2),norm_from(state,prefix+'.tcn.1'))
  self.skip=nn.Identity()
  if prefix+'.residual.0.weight' in state:self.skip=nn.Sequential(conv_from(state,prefix+'.residual.0'),norm_from(state,prefix+'.residual.1'))
 def forward_parts(self,x):
  residual=self.skip(x);a=self.graph(x);b,kc,t,v=a.shape
  a=a.reshape(b,self.partitions,kc//self.partitions,t,v)
  a=torch.einsum('bkctv,kvw->bctw',a,self.adjacency)
  a=torch.relu(self.graph_norm(a));return torch.relu(self.temporal(a)+residual),a,residual
 def forward(self,x):return self.forward_parts(x)[0]

class PoseEncoder(nn.Module):
 def __init__(self,state,include_face=True):
  super().__init__();self.parts=['body','left','right']+(['face_all'] if include_face else []);self.projection=nn.ModuleDict();self.spatial=nn.ModuleDict();self.temporal=nn.ModuleDict()
  for part in ['body','right']+(['face_all'] if include_face else []):
   layer=nn.Linear(3,64)
   with torch.no_grad():layer.weight.copy_(state['proj_linear.'+part+'.weight'].float());layer.bias.copy_(state['proj_linear.'+part+'.bias'].float())
   self.projection[part]=layer
   self.spatial[part]=nn.Sequential(*[GraphBlock(state,f'gcn_modules.{part}.layer{i}_0') for i in range(3)])
   self.temporal[part]=nn.Sequential(*[GraphBlock(state,f'fusion_gcn_modules.{part}.layer0_{i}') for i in range(3)])
  self.projection['left']=self.projection['right'];self.spatial['left']=self.spatial['right'];self.temporal['left']=self.temporal['right']
 def forward(self,body,left,right,face=None):
  inputs=[body,left,right]+([face] if len(self.parts)==4 else []);features=[];tails=[];body_feature=None
  for part,x in zip(self.parts,inputs):
   x=self.projection[part](x).permute(0,3,1,2);x=self.spatial[part](x)
   if part=='body':body_feature=x
   else:
    anchor={'left':-2,'right':-1,'face_all':0}[part];x=x+body_feature[:,:,:,anchor:anchor+1] if anchor>=0 else x+body_feature[:,:,:,anchor].unsqueeze(-1)
   for block in self.temporal[part][:-1]:x=block(x)
   output,graph,residual=self.temporal[part][-1].forward_parts(x)
   features.append(output.mean(-1));tails.append(torch.cat((graph.mean(-1),residual.mean(-1)),dim=1))
  return torch.cat(features,dim=1),torch.cat(tails,dim=1)
