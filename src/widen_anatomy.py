"""Function-preserving interleaved channel widening of our anatomy model."""
import argparse
from pathlib import Path
import torch
from echogesture.temporal3d import Temporal3D


def widen(state):
    if state['architecture']!='anatomy' or state['classifier']!='cosine':
        raise ValueError('Expected our anatomy model with cosine classifier')
    old=Temporal3D(len(state['labels']),state['channels'],state['frames'],308,'cosine','anatomy')
    old.load_state_dict(state['model']);old.eval()
    new=Temporal3D(len(state['labels']),2*state['channels'],state['frames'],308,'cosine','anatomy')
    copied={}
    for key,value in state['model'].items():
        target=new.state_dict()[key]
        if key=='head.2.weight':
            expanded=value.repeat_interleave(2,dim=1)/2
        elif '.depthwise.weight' in key:
            expanded=value.repeat_interleave(2,dim=0)
        elif value.ndim==1:
            expanded=value.repeat_interleave(2)
        elif value.ndim==2:
            expanded=value.repeat_interleave(2,dim=0)
            if target.shape[1]==2*value.shape[1]:expanded=expanded.repeat_interleave(2,dim=1)/2
        else:raise ValueError(f'Unexpected parameter: {key}')
        if expanded.shape!=target.shape:raise ValueError(f'Incompatible parameter: {key}')
        copied[key]=expanded
    new.load_state_dict(copied);new.eval()
    torch.manual_seed(17);x=torch.randn(3,308,state['frames'])
    with torch.inference_mode():
        before=old(x);after=new(x)
    torch.testing.assert_close(before,after,atol=3e-5,rtol=1e-5)
    result={k:v for k,v in state.items() if k not in ('optimizer','scheduler','rng_torch','history')}
    result.update(model=new.state_dict(),channels=2*state['channels'],
                  widening_max_absolute_error=float((before-after).abs().max()))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    result=widen(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);torch.save(result,args.output)
    print({'channels':result['channels'],'maximum_equivalence_error':result['widening_max_absolute_error']})
