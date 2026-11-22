import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from evaluate import validate_predictions, write_results
from echogesture.features import prepare
from echogesture.conversation import Conversation
from analyze_device import analyze


def example():
    return dict(clip_id='organizer-001', prediction='HELP', top5=['HELP', 'HOME'], confidence=0.4, latency_ms=10)


def test_complete_manifest_with_exact_id():
    validate_predictions([example()], [{'clip_id': 'organizer-001'}], ['HELP', 'HOME'])
    with pytest.raises(ValueError):
        validate_predictions([example(), example()], [{'clip_id': 'organizer-001'}], ['HELP', 'HOME'])
    with pytest.raises(ValueError):
        validate_predictions([example()], [{'clip_id': 'other'}], ['HELP', 'HOME'])


@pytest.mark.parametrize('key,value', [('confidence', float('nan')), ('latency_ms', -1),
    ('confidence', True), ('prediction', 'help'), ('top5', ['HOME', 'HELP']), ('top5', ['HELP', 'HELP'])])
def test_reject_invalid_predictions(key, value):
    row = example()
    row[key] = value
    with pytest.raises(ValueError):
        validate_predictions([row], [{'clip_id': 'organizer-001'}], ['HELP', 'HOME'])


def test_missing_measurements_never_pass(tmp_path):
    import csv
    path = tmp_path / 'results.csv'
    write_results(path, {'model_size': 20.0, 'latency_p95': 50.0}, 'test only', 'test only')
    rows = {r['metric']: r for r in csv.DictReader(path.open())}
    assert rows['model_size']['pass'] == 'FAIL'
    assert rows['latency_p95']['pass'] == 'FAIL'
    assert rows['top1_accuracy']['pass'] == 'NOT_MEASURED'


def test_normalization_translation_scale_and_zero_input():
    rng = np.random.default_rng(5)
    points = rng.uniform(0.2, 0.8, (12, 75, 3)).astype(np.float32)
    np.testing.assert_allclose(prepare(points), prepare(points * 2 + 3), atol=1e-5)
    assert np.isfinite(prepare(np.zeros((3, 75, 3), dtype=np.float32))).all()
    assert prepare(points).shape == (1, 2, 12, 27)


def test_confirm_and_undo_preserve_user_control():
    state = Conversation()
    state.propose(example())
    assert not state.caption
    assert state.confirm(1)
    assert state.caption == ['HOME']
    assert not state.confirm(0)
    state.undo()
    assert state.caption == []
    state.propose(example(), visible=False)
    assert state.pending is None
    state.clear()
    assert state.pending is None and state.caption == []


def test_short_phone_run_cannot_claim_thermal_or_power_pass():
    report = analyze({'samples': [dict(elapsed_s=i, landmark_stage_ms=20, classifier_ms=10) for i in range(1, 40)]})
    assert report['thermal_drift_10min'] is None
    assert report['avg_power'] is None
    assert report['compute_p95_ms'] == 30
    with pytest.raises(ValueError):
        analyze({'samples': []})


def test_full_thermal_window_detects_slowdown():
    samples = [dict(elapsed_s=i, landmark_stage_ms=20 if i < 540 else 30, classifier_ms=10) for i in range(1, 602)]
    report = analyze({'samples': samples})
    assert report['thermal_drift_10min'] == pytest.approx(100 / 3)
