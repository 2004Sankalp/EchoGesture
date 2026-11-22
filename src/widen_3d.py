"""Double attention width while preserving the source evaluation function.

Only own validation-selected checkpoints are transferred. A real training
tensor checks numerical equivalence before the wider initialization is saved.
"""
import argparse
from pathlib import Path
import json
import torch
from echogesture.temporal3d import Temporal3D


def widen(state):
    if state.get('architecture')!='attention' or state.get('classifier')!='cosine':
        raise ValueError('Widening supports attention models with cosine heads')
    channels=state['channels'];weights=state['model'];expanded={}
    order=torch.cat((torch.arange(channels),torch.arange(channels),
                     torch.arange(channels,2*channels),torch.arange(channels,2*channels)))
    for key,value in weights.items():
        if key=='stem.weight':new=value.repeat(2,1)
        elif key=='stem.bias':new=value.repeat(2)
        elif key=='positions':new=value.repeat(1,1,2)
        elif key=='head.2.weight':new=value[:,order]
        elif key.startswith('head.0.'):new=value[order]
        elif key.endswith('self_attn.in_proj_weight'):
            new=torch.cat([part.repeat(2,2)/2 for part in value.chunk(3,dim=0)],dim=0)
        elif key.endswith('self_attn.in_proj_bias'):
            new=torch.cat([part.repeat(2) for part in value.chunk(3)],dim=0)
        elif value.ndim==2:new=value.repeat(2,2)/2
        elif value.ndim==1:new=value.repeat(2)
        else:raise ValueError(f'Unexpected tensor: {key}')
        expanded[key]=new
    result={key:value for key,value in state.items() if key not in ('optimizer','scheduler','history','rng_torch')}
    result['model']=expanded;result['channels']=channels*2
    result['initialization']='Function-preserving width doubling of own validation-selected 3D model'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);larger=widen(state)
    kwargs=dict(classes=len(state['labels']),frames=state['frames'],input_features=state['model']['stem.weight'].shape[1],classifier='cosine',architecture='attention')
    original=Temporal3D(channels=state['channels'],**kwargs);original.load_state_dict(state['model']);original.eval()
    target=Temporal3D(channels=larger['channels'],**kwargs);target.load_state_dict(larger['model']);target.eval()
    root=Path(args.data);manifest=json.loads((root/'manifest.json').read_text());train=next(r for r in manifest if r['split']=='train')
    import numpy as np
    sample=torch.from_numpy(np.load(root/train['features'],allow_pickle=False).astype(np.float32)[None])
    with torch.inference_mode():a=original(sample);b=target(sample)
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    larger['width_rewrite_max_abs_error']=float((a-b).abs().max())
    torch.save(larger,args.output)
    print({'channels':larger['channels'],'max_abs_error':larger['width_rewrite_max_abs_error'],'parameters':sum(p.numel() for p in target.parameters())})


if __name__=='__main__':main()
