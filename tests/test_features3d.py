import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest
from echogesture.features3d import canonical_hand, describe, prepare3d
from prepare_3d_dataset import clean_splits


def hand():
    rng = np.random.default_rng(39)
    return rng.normal(size=(5, 21, 3)).astype(np.float32)


def test_canonical_hand_is_rigid_rotation_translation_scale_invariant():
    points = hand()
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = 0.71
    rotation = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
    original, _, valid = canonical_hand(points)
    transformed, _, valid2 = canonical_hand((points @ rotation.T)*3 + [4, -7, 2])
    assert np.all(valid == 1) and np.all(valid2 == 1)
    np.testing.assert_allclose(original, transformed, atol=2e-6)


def test_depth_changes_are_observable():
    points = hand()
    first = canonical_hand(points)[0]
    points[:, 8, 2] += 2
    second = canonical_hand(points)[0]
    assert not np.allclose(first, second)


def test_body_relative_features_cancel_common_camera_roll():
    rng=np.random.default_rng(9)
    points=rng.uniform(0.2,0.8,(8,75,3)).astype(np.float32)
    points[:,11,:2]=[0.7,0.4];points[:,12,:2]=[0.3,0.4]
    angle=0.4
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                       [np.sin(angle),np.cos(angle),0],[0,0,1]],dtype=np.float32)
    transformed=points@rotation.T
    np.testing.assert_allclose(describe(points),describe(transformed),atol=2e-5)


def test_missing_or_collinear_hand_is_masked():
    shape, basis, valid = canonical_hand(np.zeros((3, 21, 3), dtype=np.float32))
    assert not shape.any() and not basis.any() and not valid.any()
    collinear = np.zeros((2, 21, 3), dtype=np.float32)
    collinear[:, :, 0] = np.arange(21)
    assert not canonical_hand(collinear)[2].any()


def test_full_features_are_finite_and_preserve_frame_count():
    points = np.zeros((9, 75, 3), dtype=np.float32)
    points[:, 33:54] = np.tile(hand()[0], (9, 1, 1))
    assert describe(points).shape == (9, 308)
    assert prepare3d(points).shape == (308, 48)
    assert np.isfinite(prepare3d(points)).all()
    with pytest.raises(ValueError):
        describe(np.zeros((0, 75, 3)))


def test_cross_split_alias_duplicates_never_enter_training():
    rows=[dict(clip_id='mother',split='train',label='MOTHER'),dict(clip_id='mom',split='test',label='MOM'),
          dict(clip_id='val-copy',split='val',label='MOM'),dict(clip_id='unique',split='train',label='HELP')]
    kept,excluded=clean_splits(rows,{'mother':'identical','mom':'identical','val-copy':'identical','unique':'different'})
    assert [r['clip_id'] for r in kept]==['mom','unique']
    assert {r['clip_id'] for r in excluded}=={'mother','val-copy'}
