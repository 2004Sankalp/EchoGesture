import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from echogesture.features3d import describe
from echogesture.augment3d import stretch_vertical


def test_coordinate_augmentation_matches_recomputed_geometry():
    rng=np.random.default_rng(4)
    points=rng.uniform(.2,.8,(5,75,3)).astype(np.float32)
    points[:,11,:]=[.7,.4,.1];points[:,12,:]=[.3,.4,.1]
    original=describe(points).T[None]
    transformed=points.copy();transformed[:,:,1]*=1.3
    expected=describe(transformed).T[None]
    actual=stretch_vertical(torch.from_numpy(original),torch.tensor([1.3])).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-5)
    np.testing.assert_array_equal(original,describe(points).T[None])


def test_coordinate_augmentation_keeps_missing_hands_missing():
    points=np.zeros((4,75,3),dtype=np.float32)
    features=torch.from_numpy(describe(points).T[None])
    output=stretch_vertical(features,torch.tensor([1.6]))
    assert torch.isfinite(output).all()
    assert not output.any()


def test_numpy_deployment_transform_matches_training():
    from echogesture.geometry import stretch_vertical as numpy_stretch
    rng=np.random.default_rng(9)
    points=rng.uniform(.2,.8,(8,75,3)).astype(np.float32)
    points[2:4,33:54]=0
    original=describe(points).T
    for factor in [.67,.8,1.,1.25,1.5]:
        expected=stretch_vertical(torch.from_numpy(original[None]),torch.tensor([factor])).numpy()[0]
        np.testing.assert_allclose(numpy_stretch(original,factor),expected,atol=2e-5)
