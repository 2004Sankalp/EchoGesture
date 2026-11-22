import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch=pytest.importorskip('torch')
from echogesture.temporal3d import Temporal3D
from echogesture.fusion import ClipFusion

def test_full_rank_clip_fusion_matches_weighted_logits_without_frame_alignment():
 torch.set_num_threads(1);torch.manual_seed(41)
 pose=Temporal3D(7,32,48,308,'cosine','anatomy').eval()
 image=Temporal3D(7,32,16,15362,'cosine','appearance').eval()
 x=torch.randn(2,308,48);x[:,[100,173]]=1
 rgb=torch.randn(2,15362,16);rgb[:,-2:]=1
 with torch.inference_mode():expected=.4*pose(x)+.6*image(rgb)
 fused=ClipFusion([pose,image],['3d','rgb'],[.4,.6],rank=7).eval()
 with torch.inference_mode():torch.testing.assert_close(fused(x,rgb),expected,atol=1e-5,rtol=1e-5)
 assert abs(fused.retained_squared_singular_energy-1)<1e-5
