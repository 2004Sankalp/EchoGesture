import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from echogesture.opencv_landmarks import assign_hands,pose_palm_proposal


def detection(x,y):
    hand=np.zeros(132,dtype=np.float32)
    hand[4:67]=np.tile([x,y,0],21)
    return hand


def test_anatomical_wrists_override_selfie_handedness_and_reject_bystander():
    points=np.zeros((75,3),dtype=np.float32)
    points[11,:2]=[.65,.3];points[12,:2]=[.35,.3]
    points[15,:2]=[.7,.5];points[16,:2]=[.3,.5]
    output=assign_hands(points,[detection(300,500),detection(700,500),detection(100,950)],1000,1000)
    np.testing.assert_allclose(output[33,:2],[.7,.5])
    np.testing.assert_allclose(output[54,:2],[.3,.5])


def test_guided_crop_cannot_overwrite_an_existing_other_hand():
    points=np.zeros((75,3),dtype=np.float32)
    points[11,:2]=[.65,.3];points[12,:2]=[.35,.3]
    points[15,:2]=[.7,.5];points[16,:2]=[.3,.5]
    points[33:54]=[.7,.5,.01]
    output=assign_hands(points,[detection(300,500)],1000,1000,fixed_side=1)
    np.testing.assert_allclose(output[33],[.7,.5,.01])
    np.testing.assert_allclose(output[54,:2],[.3,.5])


def test_uncertain_or_degenerate_pose_does_not_create_a_crop():
    body=np.zeros((39,5),dtype=np.float32)
    assert pose_palm_proposal(body,0) is None
