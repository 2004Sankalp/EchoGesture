import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echogesture.citizen import to_mediapipe75


def test_original_holistic_units_hand_order_and_missingness():
    components=[SimpleNamespace(name=name,points=list(range(count))) for name,count in
        [('POSE_LANDMARKS',33),('FACE_LANDMARKS',468),('LEFT_HAND_LANDMARKS',21),
         ('RIGHT_HAND_LANDMARKS',21),('POSE_WORLD_LANDMARKS',33)]]
    data=np.ones((2,1,576,3),np.float32)
    data[:,:,0]=[320,240,-.5]
    data[:,:,501]=[160,120,-.1]
    data[:,:,522]=[480,360,-.2]
    confidence=np.ones((2,1,576),np.float32);confidence[1,0,501]=0
    pose=SimpleNamespace(header=SimpleNamespace(components=components,dimensions=SimpleNamespace(width=640,height=480)),
        body=SimpleNamespace(data=np.ma.array(data),confidence=confidence))
    result=to_mediapipe75(pose)
    assert result.shape==(2,75,3)
    np.testing.assert_allclose(result[0,0],[.5,.5,-.5])
    np.testing.assert_allclose(result[0,33],[.25,.25,-.1])
    np.testing.assert_allclose(result[0,54],[.75,.75,-.2])
    np.testing.assert_array_equal(result[1,33],[0,0,0])
