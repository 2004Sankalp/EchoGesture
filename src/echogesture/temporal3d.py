"""Small temporal/channel MLP mixer for CPU training and edge conversion."""
import torch
from torch import nn
import math
from .features3d import FEATURE_DIM


class CosineClassifier(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(classes,features))
        nn.init.normal_(self.weight,std=0.02)

    def forward(self,x):
        return 20 * torch.nn.functional.linear(torch.nn.functional.normalize(x,dim=-1),
                                               torch.nn.functional.normalize(self.weight,dim=-1))


class MixerBlock(nn.Module):
    def __init__(self, channels, frames):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.channel_norm = nn.LayerNorm(channels)
        self.token = nn.Sequential(nn.Linear(frames, frames), nn.ReLU(), nn.Linear(frames, frames), nn.Dropout(0.1))
        self.channel = nn.Sequential(nn.Linear(channels, channels*2), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(channels*2, channels), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.token(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel(self.channel_norm(x))


class MotionBlock(nn.Module):
    """Local motion at several time scales, with clip-independent normalization."""
    def __init__(self, channels, dilation, stride=1):
        super().__init__()
        self.norm=nn.GroupNorm(8,channels)
        self.depthwise=nn.Conv1d(channels,channels,5,padding=2*dilation,
                                 dilation=dilation,stride=stride,groups=channels)
        self.dilation,self.stride=dilation,stride
        self.mix=nn.Sequential(nn.Linear(channels,channels*2),nn.GELU(),
                               nn.Dropout(.15),nn.Linear(channels*2,channels))
        self.skip=nn.AvgPool1d(stride,stride) if stride>1 else nn.Identity()

    def forward(self,x):
        # Explicit depthwise filtering avoids extremely slow grouped Conv1d
        # backward kernels on the CPU-only Mac, while retaining the operation.
        normalized=nn.functional.pad(self.norm(x),(2*self.dilation,2*self.dilation))
        local=sum(normalized[:,:,i*self.dilation:i*self.dilation+x.shape[-1]:self.stride]
                  *self.depthwise.weight[:,0,i][None,:,None] for i in range(5))
        local=local+self.depthwise.bias[None,:,None]
        return self.skip(x)+self.mix(local.transpose(1,2)).transpose(1,2)


class AnatomyTemporal(nn.Module):
    """Shared hand-shape encoder, explicit motion stream and temporal convolutions."""
    def __init__(self,channels,input_features=308):
        super().__init__()
        self.hand=nn.Sequential(nn.Linear(63,channels//2),nn.LayerNorm(channels//2),nn.GELU(),
                                nn.Linear(channels//2,channels//2))
        self.hands=nn.Linear(channels,channels)
        self.motion=nn.Linear(input_features,channels,bias=False)
        self.blocks=nn.Sequential(MotionBlock(channels,1),MotionBlock(channels,2),
                                  MotionBlock(channels,1,2),MotionBlock(channels,2),
                                  MotionBlock(channels,1,2),MotionBlock(channels,2))

    def forward(self,x,features):
        left=self.hand(features[:,28:91].transpose(1,2))
        right=self.hand(features[:,101:164].transpose(1,2))
        left=left*features[:,100:101].transpose(1,2)
        right=right*features[:,173:174].transpose(1,2)
        motion=features[:,:,1:]-features[:,:,:-1]
        motion=nn.functional.pad(motion,(1,0))
        x=x+self.hands(torch.cat((left,right),dim=-1))+.25*self.motion(motion.transpose(1,2))
        return self.blocks(x.transpose(1,2)).transpose(1,2)


class GenericMotionTemporal(nn.Module):
    """Temporal motion branch for independent landmark feature profiles."""
    def __init__(self, channels, input_features):
        super().__init__()
        self.motion=nn.Linear(input_features,channels,bias=False)
        self.blocks=nn.Sequential(MotionBlock(channels,1),MotionBlock(channels,2),
                                  MotionBlock(channels,1,2),MotionBlock(channels,2),
                                  MotionBlock(channels,1,2),MotionBlock(channels,2))

    def forward(self,x,features):
        velocity=nn.functional.pad(features[:,:,1:]-features[:,:,:-1],(1,0))
        x=x+.25*self.motion(velocity.transpose(1,2))
        return self.blocks(x.transpose(1,2)).transpose(1,2)


class AppearanceProjection(nn.Module):
    """Learn a shared descriptor for four patches in each of three RGB views."""
    def __init__(self):
        super().__init__()
        self.patch=nn.Sequential(nn.Linear(1280,64),nn.LayerNorm(64),nn.GELU())

    def forward(self,x):
        local=x[:,:15360].transpose(1,2).reshape(x.shape[0],x.shape[-1],3,4,1280)
        mask=torch.cat((torch.ones_like(x[:,-2:-1]),x[:,-2:].clamp(0,1)),dim=1).transpose(1,2)
        encoded=self.patch(local)*mask[:,:,:,None,None]
        encoded=encoded.reshape(x.shape[0],x.shape[-1],768).transpose(1,2)
        return torch.cat((encoded,x[:,-2:].clamp(0,1)),dim=1)


class Temporal3D(nn.Module):
    def __init__(self, classes=2000, channels=128, frames=48, input_features=FEATURE_DIM, classifier='linear', architecture='mixer'):
        super().__init__()
        if architecture=='appearance':
            if input_features!=15362:raise ValueError('Expected three four-patch 1280-channel views plus two masks')
            self.appearance=AppearanceProjection()
            input_features=770
        self.stem = nn.Linear(input_features+(46 if architecture=='anatomy_kinematic' else 0), channels)
        self.architecture=architecture
        if architecture=='attention':
            if channels%32:raise ValueError('Attention channels must be a multiple of 32')
            layer=nn.TransformerEncoderLayer(channels,channels//32,channels*2,dropout=0.2,activation='gelu',batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            position=torch.arange(frames).float().unsqueeze(1)
            divisor=torch.exp(torch.arange(0,channels,2).float()*(-math.log(10000.0)/channels))
            encoding=torch.zeros(1,frames,channels)
            encoding[0,:,0::2]=torch.sin(position*divisor)
            encoding[0,:,1::2]=torch.cos(position*divisor)
            self.register_buffer('positions',encoding)
        elif architecture in ('anatomy','anatomy_context'):
            self.temporal=AnatomyTemporal(channels,input_features)
            if architecture=='anatomy_context':
                self.context=nn.TransformerEncoderLayer(channels,4,channels*2,dropout=.2,
                    activation='gelu',batch_first=True,norm_first=True)
                self.context_gate=nn.Parameter(torch.zeros(()))
        elif architecture=='anatomy_kinematic':self.temporal=AnatomyTemporal(channels,input_features+46)
        elif architecture=='motion':self.temporal=GenericMotionTemporal(channels,input_features)
        elif architecture=='appearance':self.temporal=GenericMotionTemporal(channels,input_features)
        else:self.temporal = nn.Sequential(*[MixerBlock(channels, frames) for _ in range(4)])
        classifier_layer=(nn.Linear(channels*2,classes) if classifier=='linear' else CosineClassifier(channels*2,classes))
        self.head = nn.Sequential(nn.LayerNorm(channels*2), nn.Dropout(0.25), classifier_layer)

    def encode(self, x):
        if self.architecture=='appearance':x=self.appearance(x)
        if self.architecture=='anatomy_kinematic':x=torch.cat((x,finger_kinematics(x)),dim=1)
        features=x
        x = self.stem(x.transpose(1, 2))
        if self.architecture=='attention':x=x+self.positions
        x = self.temporal(x,features) if self.architecture in ('anatomy','anatomy_context','anatomy_kinematic','motion','appearance') else self.temporal(x)
        if self.architecture=='anatomy_context':x=x+self.context_gate.tanh()*self.context(x)
        return torch.cat((x.mean(1), x.amax(1)), dim=1)

    def forward(self,x):
        return self.head(self.encode(x))


def finger_kinematics(features):
    """46 rigid/scale invariant bend, spread and pinch channels; missing=zero."""
    result=[]
    for start,valid in ((28,100),(101,173)):
        q=features[:,start:start+63].reshape(features.shape[0],21,3,features.shape[-1])
        chains=q[:,[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],
                    [0,13,14,15,16],[0,17,18,19,20]]]
        bones=chains[:,:,1:]-chains[:,:,:-1]
        directions=nn.functional.normalize(bones,dim=3,eps=1e-5)
        bends=(directions[:,:,:-1]*directions[:,:,1:]).sum(3).flatten(1,2)
        proximal=directions[:,:,0]
        spread=(proximal[:,:-1]*proximal[:,1:]).sum(2)
        scale=.5*((q[:,5]-q[:,17]).square().sum(1).clamp_min(1e-10).sqrt()+
                   (q[:,9]-q[:,0]).square().sum(1).clamp_min(1e-10).sqrt())
        pinch=(q[:,[8,12,16,20]]-q[:,4:5]).square().sum(2).clamp_min(1e-10).sqrt()/scale[:,None].clamp_min(1e-5)
        result.append(torch.cat((bends,spread,pinch.clamp_max(10)),dim=1)*features[:,valid:valid+1])
    return torch.cat(result,dim=1)
