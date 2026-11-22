"""Fresh word classifier over independent pose embeddings and real 3D geometry."""
import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from echogesture.temporal3d import CosineClassifier

class GeometryPoseHead(nn.Module):
 def __init__(self,pose_features=4096,channels=128,classes=2000):
  super().__init__();self.pose_features=pose_features
  self.pose_norm=nn.LayerNorm(pose_features);self.geometry_norm=nn.LayerNorm(1016)
  self.embedding=nn.Sequential(nn.Linear(pose_features+1016,channels),nn.LayerNorm(channels),nn.GELU(),nn.Dropout(.35),nn.Linear(channels,channels),nn.GELU(),nn.Dropout(.25))
  self.classifier=CosineClassifier(channels,classes)
 def encode(self,x):return self.embedding(torch.cat((self.pose_norm(x[:,:self.pose_features]),self.geometry_norm(x[:,self.pose_features:])),dim=1))
 def forward(self,x):return self.classifier(self.encode(x))
