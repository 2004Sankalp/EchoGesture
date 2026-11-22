"""Public pose-only diagnostic. Never substitutes for raw-video/phone results."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', required=True)
    parser.add_argument('--models', default='models')
    parser.add_argument('--output', default='results/public_pose_diagnostic.json')
    args = parser.parse_args()
    poses, models = Path(args.poses), Path(args.models)
    labels = json.loads((models / 'labels.json').read_text())
    manifest = json.loads((poses / 'manifest.json').read_text())
    rows = [r for r in manifest if r['split'] == 'test']
    if not rows:
        raise ValueError('No test poses')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    ort.disable_telemetry_events()
    report = dict(scope='Convenience sample from partial public WLASL pose archive. Not organiser test data, not raw-video accuracy, not phone latency.',
                  test_clips=len(rows), calibration_train_clips=sum(r['split']=='train' for r in manifest),
                  device=platform.platform(), runtime=ort.__version__, results={})
    for model in ('recognizer.fp32.onnx', 'recognizer.int8.onnx'):
        session = ort.InferenceSession(str(models / model), options, providers=['CPUExecutionProvider'])
        predictions, timings = [], []
        for row in rows:
            x = np.load(poses / 'test' / (row['clip_id'] + '.npy'), allow_pickle=False)
            start = time.perf_counter_ns()
            y = session.run(None, {'poses': x})[0][0]
            timings.append((time.perf_counter_ns() - start) / 1e6)
            ranks = np.argsort(-y)[:5]
            predictions.append(dict(clip_id=row['clip_id'], truth=row['label'], prediction=labels[int(ranks[0])],
                                    top5=[labels[int(i)] for i in ranks]))
        report['results'][model] = dict(top1_accuracy=100*sum(p['truth']==p['prediction'] for p in predictions)/len(rows),
            top5_accuracy=100*sum(p['truth'] in p['top5'] for p in predictions)/len(rows),
            classifier_p95_ms=float(np.percentile(timings,95)), model_bytes=(models/model).stat().st_size,
            predictions=predictions)
    report['quantization_accuracy_drop_pt'] = report['results']['recognizer.fp32.onnx']['top1_accuracy'] - report['results']['recognizer.int8.onnx']['top1_accuracy']
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({**report,'results':{k:{a:b for a,b in v.items() if a!='predictions'} for k,v in report['results'].items()}}, indent=2))


if __name__ == '__main__':
    main()
