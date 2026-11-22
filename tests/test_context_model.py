import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch=pytest.importorskip('torch')
from echogesture.temporal3d import Temporal3D

def test_zero_gate_transfer_preserves_all_class_predictions_and_learns_gate():
 torch.set_num_threads(1);torch.manual_seed(17)
 old=Temporal3D(2000,64,48,308,'cosine','anatomy').eval()
 new=Temporal3D(2000,64,48,308,'cosine','anatomy_context').eval()
 missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
 assert not unexpected and all(k.startswith('context') for k in missing)
 x=torch.randn(2,308,48);x[:,[100,173]]=1
 torch.testing.assert_close(old(x),new(x),atol=1e-6,rtol=1e-6)
 new(x).square().mean().backward()
 assert torch.isfinite(new.context_gate.grad) and new.context_gate.grad.abs()>0
