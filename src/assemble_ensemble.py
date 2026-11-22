"""Assemble the frozen validation-selected deployment recipe from fresh exports."""
import argparse,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--mixer',required=True);p.add_argument('--extra',required=True);p.add_argument('--wide',required=True);p.add_argument('--output',default='models/ensemble');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);members=[];labels=None
    for name,source,weight,scales in [('mixer_v2',args.mixer,.3,[.67,1.,1.5]),('extra_v3',args.extra,.4,[.67,1.,1.5]),('wide_v2',args.wide,.3,[1.])]:
        source=Path(source);info=json.loads((source/'conversion3d.json').read_text());vocabulary=json.loads((source/'labels.json').read_text())
        if labels is None:labels=vocabulary
        if vocabulary!=labels or len(labels)!=2000:raise ValueError('Must retain the identical full 2000-class label ordering')
        target=name+'.int8.onnx';shutil.copy2(source/'recognizer3d.int8.onnx',out/target);shutil.copy2(source/'conversion3d.json',out/(name+'.conversion.json'))
        members.append(dict(name=name,model=target,weight=weight,scales=scales,feature_version=info['feature_version'],frames=info['frames']))
    (out/'labels.json').write_text(json.dumps(labels,indent=2));(out/'ensemble.json').write_text(json.dumps({'selection':'Frozen validation-selected global recipe. Do not tune on final test.','members':members},indent=2))
if __name__=='__main__':main()
