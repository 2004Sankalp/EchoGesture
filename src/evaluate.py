"""Strict local prediction validation and honest measurement export.

Manifest JSONL adapter: clip_id, video_path, optional label. This is our adapter,
not an invented claim about the unavailable organiser manifest format.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def read_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row['clip_id'] for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Manifest requires unique nonempty string clip IDs')
    for row in rows:
        video = Path(row['video_path'])
        row['video_path'] = str(video if video.is_absolute() else path.parent / video)
    return rows


def validate_predictions(predictions, manifest, vocabulary):
    allowed = set(vocabulary)
    expected = {row['clip_id'] for row in manifest}
    actual = [row.get('clip_id') for row in predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Prediction IDs must match every manifest ID exactly once')
    for row in predictions:
        word = row.get('prediction')
        if not isinstance(word, str) or word != word.upper() or word not in allowed:
            raise ValueError('Prediction is not an uppercase vocabulary entry')
        for key in ('confidence', 'latency_ms'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid {key}')
        if not 0 <= row['confidence'] <= 1 or row['latency_ms'] < 0:
            raise ValueError('Confidence or latency out of range')
        if 'top5' in row:
            top = row['top5']
            if (not isinstance(top, list) or not 1 <= len(top) <= 5
                    or any(not isinstance(w, str) or w not in allowed for w in top)
                    or len(set(top)) != len(top) or top[0] != word):
                raise ValueError('Invalid top5')


TARGETS = [
    ('top1_accuracy', '>=70', '%', lambda x: x >= 70),
    ('latency_p95', '<50', 'ms', lambda x: x < 50),
    ('model_size', '<20', 'MB', lambda x: x < 20),
    ('accuracy_drop_robustness', '<=5', 'pt', lambda x: x <= 5),
    ('avg_power', '<=500', 'mW', lambda x: x <= 500),
    ('thermal_drift_10min', '<=20', '%', lambda x: x <= 20),
]


def write_results(path, measured, device, runtime):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['metric', 'target', 'measured', 'unit', 'device', 'runtime', 'pass'])
        writer.writeheader()
        for name, target, unit, check in TARGETS:
            value = measured.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid measurement: {name}')
            writer.writerow(dict(metric=name, target=target, measured='' if value is None else value,
                                 unit=unit, device=device, runtime=runtime,
                                 **{'pass': 'NOT_MEASURED' if value is None else 'PASS' if check(value) else 'FAIL'}))


def main():
    from echogesture.engine import create_recognizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--device', required=True)
    parser.add_argument('--runtime', default='ONNX Runtime CPU INT8 weights (mixed graph)')
    parser.add_argument('--backend', choices=['baseline', '3d', 'ensemble'], default='ensemble')
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    vocabulary = json.loads(Path(args.vocabulary).read_text())
    recognizer = create_recognizer(args.backend, vocabulary=args.vocabulary)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    predictions, diagnostics = [], []
    for row in manifest:
        result = recognizer.predict(row['video_path'], row['clip_id'])
        diagnostics.append({'clip_id': row['clip_id'], 'pipeline_ms': result.pop('pipeline_ms')})
        predictions.append(result)
    validate_predictions(predictions, manifest, vocabulary)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(p, allow_nan=False) + '\n' for p in predictions))
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2, allow_nan=False))
    (out / 'pipeline_timings.json').write_text(json.dumps(diagnostics, indent=2))
    measured = {}
    labels = {row['clip_id']: row['label'] for row in manifest if 'label' in row}
    if len(labels) == len(manifest):
        if any(label not in vocabulary for label in labels.values()):
            raise ValueError('Ground-truth label outside vocabulary')
        measured['top1_accuracy'] = 100 * sum(p['prediction'] == labels[p['clip_id']] for p in predictions) / len(predictions)
    # Per-clip classifier timing only. Phone steady-state benchmark is separate.
    measured['latency_p95'] = float(np.percentile([p['latency_ms'] for p in predictions], 95))
    measured['model_size'] = getattr(recognizer, 'model_asset_bytes', recognizer.model_path.stat().st_size) / 1e6
    write_results(out / 'results.csv', measured, args.device, args.runtime)
    print('Predictions written. Device, timing scope and evidence still need submission review.')


if __name__ == '__main__':
    main()
