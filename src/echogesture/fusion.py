"""Clip-level fusion with a compact, fixed low-rank classifier.

Each stream keeps its own temporal sampling. Only clip identities are joined.
Full rank reproduces the weighted cosine-logit ensemble. Lower ranks are
approximations and must be measured, including actual INT8 conversion loss.
"""
import copy
import torch
from torch import nn
from .temporal3d import CosineClassifier


class ClipFusion(nn.Module):
    def __init__(self,models,sources,weights,rank):
        super().__init__()
        if len(models)!=len(sources) or len(models)!=len(weights):raise ValueError('Fusion members differ')
        if any(s not in ('3d','rgb') for s in sources):raise ValueError('Unknown stream')
        if any(w<=0 for w in weights) or abs(sum(weights)-1)>1e-6:raise ValueError('Expected positive normalized weights')
        matrices=[];norms=[];classes=None
        for model,weight in zip(models,weights):
            if not isinstance(model.head[2],CosineClassifier):raise ValueError('Expected cosine classifiers')
            matrix=nn.functional.normalize(model.head[2].weight.detach(),dim=1)*weight
            if classes is not None and len(matrix)!=classes:raise ValueError('Class counts differ')
            classes=len(matrix);matrices.append(matrix);norms.append(copy.deepcopy(model.head[0]))
        matrix=torch.cat(matrices,dim=1)
        if not 0<rank<=min(matrix.shape):raise ValueError('Invalid fusion rank')
        with torch.no_grad():
            u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
            scale=s[:rank].clamp_min(0).sqrt()
        self.projection=nn.Linear(matrix.shape[1],rank,bias=False)
        self.classifier=nn.Linear(rank,classes,bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(scale[:,None]*vh[:rank])
            self.classifier.weight.copy_(u[:,:rank]*scale[None])
        for model in models:model.head=nn.Identity()
        self.branches=nn.ModuleList(models);self.norms=nn.ModuleList(norms)
        self.sources=sources
        self.retained_squared_singular_energy=float((s[:rank].square().sum()/s.square().sum()).item())

    def forward(self,features3d,appearance):
        vectors=[]
        for model,norm,source in zip(self.branches,self.norms,self.sources):
            encoded=model.encode(features3d if source=='3d' else appearance)
            vectors.append(nn.functional.normalize(norm(encoded),dim=1))
        return 20*self.classifier(self.projection(torch.cat(vectors,dim=1)))
