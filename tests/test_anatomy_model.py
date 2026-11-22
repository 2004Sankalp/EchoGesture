"""Guard the CPU convolution rewrite and new model's gradient path."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch=pytest.importorskip('torch')
from echogesture.temporal3d import MotionBlock,Temporal3D


def test_depthwise_cpu_rewrite_matches_forward_and_input_gradient():
    torch.set_num_threads(1)
    for dilation,stride in [(1,1),(2,1),(2,2)]:
        block=MotionBlock(32,dilation,stride).eval()
        x=torch.randn(2,32,48,requires_grad=True)
        actual=block(x)
        expected=block.skip(x)+block.mix(block.depthwise(block.norm(x)).transpose(1,2)).transpose(1,2)
        torch.testing.assert_close(actual,expected)
        first=torch.autograd.grad(actual.square().mean(),x,retain_graph=True)[0]
        second=torch.autograd.grad(expected.square().mean(),x)[0]
        torch.testing.assert_close(first,second)


def test_anatomy_model_retains_all_classes_and_finite_gradients_with_missing_hands():
    model=Temporal3D(2000,64,48,308,'cosine','anatomy')
    x=torch.zeros(2,308,48)
    y=model(x)
    assert y.shape==(2,2000)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
