import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch=pytest.importorskip('torch')
from echogesture.temporal3d import Temporal3D,finger_kinematics

def test_kinematics_invariant_to_rigid_transform_and_scale():
 torch.set_num_threads(1);torch.manual_seed(7)
 x=torch.randn(2,308,48);x[:,[100,173]]=1;y=x.clone()
 r,_=torch.linalg.qr(torch.randn(3,3))
 for start in [28,101]:
  q=x[:,start:start+63].reshape(2,21,3,48)
  y[:,start:start+63]=(torch.einsum('ij,bkjt->bkit',r,q)*1.7+2).reshape(2,63,48)
 actual=finger_kinematics(x)
 assert actual.shape==(2,46,48)
 torch.testing.assert_close(actual,finger_kinematics(y),rtol=1e-4,atol=1e-5)
 assert not finger_kinematics(torch.zeros_like(x)).any()

def test_zero_padded_initialization_preserves_predictions():
 torch.manual_seed(11)
 old=Temporal3D(2000,64,48,308,'cosine','anatomy').eval()
 new=Temporal3D(2000,64,48,308,'cosine','anatomy_kinematic').eval()
 state=old.state_dict()
 for key in ['stem.weight','temporal.motion.weight']:
  state[key]=torch.nn.functional.pad(state[key],(0,46))
 new.load_state_dict(state)
 x=torch.randn(2,308,48);x[:,[100,173]]=1
 with torch.inference_mode():torch.testing.assert_close(old(x),new(x),rtol=1e-5,atol=1e-5)
