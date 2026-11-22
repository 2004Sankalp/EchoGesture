import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echogesture.features2d import prepare2d,prepare_transfer2d,FEATURE_DIM

def test_local_geometry_is_translation_scale_and_roll_invariant():
 rng=np.random.default_rng(27);p=rng.uniform(.2,.8,(5,133,2)).astype(np.float32);s=np.ones((5,133),np.float32)
 a=.4;r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 expected=prepare2d(p,s);actual=prepare2d((p@r.T)*1.3+2,s)
 assert expected.shape==(FEATURE_DIM,48)
 np.testing.assert_allclose(actual,expected,atol=2e-5)

def test_invalid_and_missing_landmarks_remain_finite_and_masked():
 p=np.full((2,133,2),np.nan,np.float32);s=np.zeros((2,133),np.float32)
 actual=prepare2d(p,s)
 assert np.isfinite(actual).all()
 assert not np.any(actual)

def test_transfer_profile_retains_original_abi_and_adds_only_face_xy_and_mask():
 rng=np.random.default_rng(29);p=rng.uniform(.2,.8,(3,133,2)).astype(np.float32);s=np.ones((3,133),np.float32)
 local=prepare2d(p,s);transfer=prepare_transfer2d(p,s)
 assert transfer.shape==(512,48)
 assert np.isfinite(transfer).all()
 np.testing.assert_array_equal(transfer[308:444],local[118:254])
 np.testing.assert_array_equal(transfer[444:],local[317:385])
 assert not np.any(prepare_transfer2d(p,np.zeros_like(s)))
