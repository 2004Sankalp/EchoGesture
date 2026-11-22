import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch=pytest.importorskip('torch')
from echogesture.temporal3d import Temporal3D,AppearanceProjection

def test_missing_hands_do_not_contribute_patch_bias_or_hidden_pixels():
 torch.set_num_threads(1);torch.manual_seed(29);p=AppearanceProjection().eval()
 x=torch.randn(2,15362,16);x[:,-2:]=0;y=x.clone();y[:,5120:15360]=50
 torch.testing.assert_close(p(x),p(y))
 encoded=p(x)
 assert encoded.shape==(2,770,16)
 assert not encoded[:,256:768].any()

def test_spatial_classifier_preserves_full_vocabulary_and_finite_gradients():
 model=Temporal3D(2000,64,16,15362,'cosine','appearance');x=torch.randn(2,15362,16);x[:,-2:]=1
 output=model(x);assert output.shape==(2,2000) and torch.isfinite(output).all()
 output.square().mean().backward()
 assert torch.isfinite(model.appearance.patch[0].weight.grad).all()
