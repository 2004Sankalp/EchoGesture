"""Evaluate a frozen candidate against all clips in the declared public split."""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import onnxruntime as ort


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--split',choices=['val','test'],default='val');parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.data);models=Path(args.models);labels=json.loads((root/'labels.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text());rows=[r for r in manifest if r['split']==args.split]
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1;ort.disable_telemetry_events()
    report=dict(scope='Public inferred-XYZ pose clips, full declared WLASL2000 vocabulary; not organiser test videos; official splits overlap signers',
        split=args.split,classes=len(labels),clips=len(rows),results={})
    for name in ['recognizer3d.fp32.onnx','recognizer3d.int8.onnx']:
        session=ort.InferenceSession(str(models/name),options,providers=['CPUExecutionProvider']);predictions=[];per_class=defaultdict(list);times=[]
        for row in rows:
            x=np.load(root/row['features'],allow_pickle=False).astype(np.float32)[None]
            start=time.perf_counter_ns();y=session.run(None,{'features3d':x})[0][0];times.append((time.perf_counter_ns()-start)/1e6)
            ranks=np.argsort(-y)[:5];correct=int(ranks[0])==row['class_id'];per_class[row['class_id']].append(correct)
            predictions.append(dict(clip_id=row['clip_id'],truth=row['label'],prediction=labels[int(ranks[0])],top5=[labels[int(i)] for i in ranks]))
        report['results'][name]=dict(top1_accuracy=100*sum(p['truth']==p['prediction'] for p in predictions)/len(rows),
            top5_accuracy=100*sum(p['truth'] in p['top5'] for p in predictions)/len(rows),
            macro_accuracy=100*float(np.mean([np.mean(v) for v in per_class.values()])),classes_with_eval_clips=len(per_class),
            classifier_p95_ms=float(np.percentile(times,95)),model_bytes=(models/name).stat().st_size,predictions=predictions)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({**report,'results':{k:{a:b for a,b in v.items() if a!='predictions'} for k,v in report['results'].items()}},indent=2))


if __name__=='__main__':main()
