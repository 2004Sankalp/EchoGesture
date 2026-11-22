import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echogesture.appearance import views,MEAN,STD

def test_missing_hands_are_zero_masked_and_full_frame_kept():
 image=np.full((120,240,3),100,np.uint8)
 body,hands,mask=views(image)
 assert body.shape==(3,160,160) and hands.shape==(2,3,96,96)
 assert np.isfinite(body).all() and not np.any(hands) and not np.any(mask)
 # Padding represents the normalization mean, so it carries no image content.
 np.testing.assert_allclose(body[:,0],0,atol=1e-6)

def test_hand_crop_uses_visible_region_and_rejects_missing_wrist():
 image=np.zeros((200,200,3),np.uint8);image[50:150,50:150]=[0,0,255]
 points=np.stack((np.linspace(.35,.65,21),np.linspace(.35,.65,21)),-1)
 _,hands,mask=views(image,np.stack((points,points)),np.ones((2,21),np.float32))
 np.testing.assert_array_equal(mask,[1,1])
 expected=(np.array([1.,0.,0.])-MEAN)/STD
 np.testing.assert_allclose(hands[0,:,48,48],expected,atol=1e-6)
 scores=np.ones((2,21),np.float32);scores[0,0]=0
 _,hands,mask=views(image,np.stack((points,points)),scores)
 assert mask[0]==0 and not np.any(hands[0])
