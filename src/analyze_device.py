"""Analyze exported phone timings; never invent power or accuracy values."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import write_results


def analyze(report):
    samples = report['samples']
    if not samples:
        raise ValueError('No benchmark samples. Run the camera benchmark first.')
    times = np.array([[s['elapsed_s'], s['landmark_stage_ms'], s['classifier_ms']] for s in samples], dtype=float)
    if not np.isfinite(times).all() or (times < 0).any() or (np.diff(times[:,0]) < 0).any():
        raise ValueError('Invalid or unordered benchmark measurements')
    combined = times[:,1:].sum(1)
    first = combined[times[:,0] <= 60]
    last = combined[(times[:,0] >= 540) & (times[:,0] <= 610)]
    drift = None
    if times[-1,0] >= 600 and len(first) >= 30 and len(last) >= 30:
        start_p95 = float(np.percentile(first,95))
        if start_p95 > 0:
            drift = max(0.0, 100 * (float(np.percentile(last,95)) / start_p95 - 1))
    return dict(sample_count=len(samples), elapsed_s=float(times[-1,0]),
        classifier_p95_ms=float(np.percentile(times[:,2],95)),
        landmark_stage_p95_ms=float(np.percentile(times[:,1],95)),
        compute_p95_ms=float(np.percentile(combined,95)), thermal_drift_10min=drift,
        scope='Combined camera conversion, landmark extraction, feature geometry and ensemble compute. Camera delivery, capture window, HUD and human confirmation delay excluded. Full user-perceived latency unverified.',
        thermal_method='p95 combined compute during final minute divided by first minute, minus one; minimum 10-minute duration. Confirm this definition with organisers.',
        avg_power=None, accuracy=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence')
    parser.add_argument('--output', default='results/device')
    args = parser.parse_args()
    raw = json.loads(Path(args.evidence).read_text())
    report = analyze(raw)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    models = Path(__file__).resolve().parents[1] / 'models'
    config = json.loads((models/'ensemble/ensemble.json').read_text())
    model_bytes = (models/'holistic_landmarker.task').stat().st_size + sum((models/'ensemble'/m['model']).stat().st_size for m in config['members'])
    measured = dict(latency_p95=report['compute_p95_ms'], thermal_drift_10min=report['thermal_drift_10min'],model_size=model_bytes/1e6)
    write_results(out/'results.csv',measured,raw['device']+' Android '+raw['android'],'MediaPipe Tasks + ONNX Runtime CPU INT8-QDQ, combined compute only')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
