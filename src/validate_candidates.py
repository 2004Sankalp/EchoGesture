"""Validation-only global model/geometry selection. No test samples are read."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from echogesture.temporal3d import Temporal3D
from echogesture.augment3d import stretch_vertical


def load_candidate(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=Temporal3D(len(state['labels']),state['channels'],state['frames'],
        state.get('input_features',state['model']['stem.weight'].shape[1]),state.get('classifier','linear'),state.get('architecture','mixer'))
    model.load_state_dict(state['model']);model.eval()
    return model,state


def metrics(logits,truth):
    top=np.argsort(-logits,axis=1)[:,:5]
    return {'top1_accuracy':100*float(np.mean(top[:,0]==truth)),
            'top5_accuracy':100*float(np.mean(np.any(top==truth[:,None],axis=1)))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    candidates=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    datasets={};all_logits=[];report={'scope':'Clean public validation only, full 2000-class outputs. Global weights only, no class-specific routing. Test not read.','models':{}}
    common_ids=truth=labels=None
    profiles={'single':[1.0],'narrow_tta':[0.8,1.0,1.25],'wide_tta':[0.67,1.0,1.5]}
    for candidate in candidates:
        root=Path(candidate['data']);key=str(root)
        if key not in datasets:
            rows=[r for r in json.loads((root/'manifest.json').read_text()) if r['split']=='val']
            x=torch.from_numpy(np.stack([np.load(root/r['features'],allow_pickle=False) for r in rows]).astype(np.float32))
            datasets[key]=(x,rows)
        x,rows=datasets[key];ids=[r['clip_id'] for r in rows]
        model,state=load_candidate(candidate['checkpoint'])
        if common_ids is None:
            common_ids=ids;truth=np.array([r['class_id'] for r in rows]);labels=state['labels']
        if common_ids!=ids or labels!=state['labels']:raise ValueError('Validation identity/vocabulary mismatch')
        scales=sorted(set(v for p in profiles.values() for v in p));scale_logits={}
        with torch.inference_mode():
            for scale in scales:
                logits=[]
                for start in range(0,len(x),64):
                    batch=x[start:start+64]
                    if scale!=1.0:batch=stretch_vertical(batch,torch.full((len(batch),),scale))
                    logits.append(model(batch).numpy())
                scale_logits[scale]=np.concatenate(logits)
        choices={name:sum(scale_logits[s] for s in scales)/len(scales) for name,scales in profiles.items()}
        scores={name:metrics(logits,truth) for name,logits in choices.items()}
        selected=max(scores,key=lambda name:scores[name]['top1_accuracy'])
        selected_logits=choices[selected];all_logits.append(selected_logits)
        np.savez(out/(candidate['name']+'.npz'),clip_ids=np.array(ids),truth=truth,logits=selected_logits.astype(np.float16))
        record={'profiles':scores,'selected_profile':selected,'feature_version':state['feature_version'],
                'checkpoint_sha256':hashlib.sha256(Path(candidate['checkpoint']).read_bytes()).hexdigest()}
        report['models'][candidate['name']]=record
        print(json.dumps({'model':candidate['name'],**record}),flush=True)
    if len(all_logits)==3:
        best=None
        for first in range(11):
            for second in range(11-first):
                weights=[first/10,second/10,(10-first-second)/10]
                combined=sum(w*v for w,v in zip(weights,all_logits))
                score=metrics(combined,truth)
                if best is None or score['top1_accuracy']>best['top1_accuracy']:
                    best={**score,'weights':weights,'models':[v['name'] for v in candidates]}
        report['selected_global_ensemble']=best
    report['validation_clips']=len(common_ids);report['output_classes']=len(labels)
    (out/'selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report.get('selected_global_ensemble',{})),flush=True)


if __name__=='__main__':main()
