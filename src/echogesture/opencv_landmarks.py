"""Explicit CPU landmark extraction, with no OpenGL or network dependency.

Uses credited OpenCV Zoo MediaPipe-converted models. It is a separate extractor
profile from the historical Holistic poses; accuracy must be measured on videos.
"""
from pathlib import Path
import cv2
import numpy as np
from vendor.opencv_zoo.mp_palmdet import MPPalmDet
from vendor.opencv_zoo.mp_handpose import MPHandPose
from vendor.opencv_zoo.mp_persondet import MPPersonDet
from vendor.opencv_zoo.mp_pose import MPPose


def assign_hands(points, hands, width, height, fixed_side=None):
    """Associate detected hands with anatomical body wrists, not selfie labels.

    Reject distant hands to reduce attaching a bystander's hand to the signer.
    This geometry rule is not a validated multi-person tracking system.
    """
    wrists=points[[15,16],:2]*[width,height]
    shoulders=points[[11,12],:2]*[width,height]
    threshold=max(30.,float(np.linalg.norm(shoulders[0]-shoulders[1]))*1.3)
    pairs=[]
    for index,hand in enumerate(hands):
        wrist=np.asarray(hand)[4:67].reshape(21,3)[0,:2]
        for side in range(2):
            if fixed_side is not None and side!=fixed_side:continue
            if np.any(points[15+side]):
                distance=float(np.linalg.norm(wrist-wrists[side]))
                if distance<=threshold:pairs.append((distance,side,index))
    used_sides={side for side in range(2) if np.any(points[33+side*21:54+side*21])};used_hands=set()
    for _,side,index in sorted(pairs):
        if side in used_sides or index in used_hands:continue
        hand=np.asarray(hands[index])[4:67].reshape(21,3).copy()
        hand[:,0]/=width;hand[:,1]/=height;hand[:,2]/=width
        offset=33 if side==0 else 54
        points[offset:offset+21]=hand
        used_sides.add(side);used_hands.add(index)
    return points


def pose_palm_proposal(body, side):
    """Propose a hand crop from body wrist/finger points, not hand output."""
    wrist=body[15+side,:2];index=body[19+side,:2];pinky=body[17+side,:2];thumb=body[21+side,:2]
    if body[15+side,3]<0.2 or body[15+side,4]<0.2:return None
    landmarks=np.stack((wrist,index,(index+pinky)/2,(index+3*pinky)/4,pinky,thumb,(thumb+wrist)/2))
    extent=np.ptp(landmarks,axis=0)
    minimum=max(5.,float(np.linalg.norm(body[11,:2]-body[12,:2]))*.1)
    if np.linalg.norm(index-wrist)<2:return None
    centre=(landmarks.min(0)+landmarks.max(0))/2
    half=np.maximum(extent,minimum)/2
    return np.r_[centre-half,centre+half,landmarks.reshape(-1),1.].astype(np.float32)


class CPUExtractor:
    def __init__(self, directory=None, pose_guidance=True):
        directory=Path(directory or Path(__file__).resolve().parents[2]/'models/opencv')
        names=['palm_detection_mediapipe_2023feb_int8bq.onnx','handpose_estimation_mediapipe_2023feb_int8bq.onnx',
               'person_detection_mediapipe_2023mar_int8bq.onnx','pose_estimation_mediapipe_2023mar_int8bq.onnx']
        for name in names:
            if not (directory/name).is_file():raise FileNotFoundError(f'Missing CPU landmark model: {directory/name}')
        cv2.setNumThreads(1)
        self.pose_guidance=pose_guidance
        self.palm=MPPalmDet(str(directory/names[0]),backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)
        self.hand=MPHandPose(str(directory/names[1]),confThreshold=0.5,backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)
        self.person=MPPersonDet(str(directory/names[2]),backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)
        self.pose=MPPose(str(directory/names[3]),backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)

    def process(self, image):
        height,width=image.shape[:2]
        points=np.zeros((75,3),dtype=np.float32)
        people=self.person.infer(image)
        if not len(people):return points
        primary=people[int(np.argmax(people[:,-1]))].copy()
        body=self.pose.infer(image,primary)
        if body is None:return points
        points[:33]=body[1][:33,:3]
        points[:33,0]/=width;points[:33,1]/=height;points[:33,2]/=width
        hands=[]
        palms=self.palm.infer(image)
        for palm in sorted(palms,key=lambda p:float(p[-1]),reverse=True)[:4]:
            try:hand=self.hand.infer(image,palm.copy())
            except cv2.error as error:
                # Empty border crops can occur for partially visible hands.
                # Unsupported model/runtime errors must still be surfaced.
                if 'resize' in str(error) or 'cvtColor' in str(error):continue
                raise
            if hand is not None:hands.append(hand)
        points=assign_hands(points,hands,width,height)
        if self.pose_guidance:
            for side in range(2):
                if np.any(points[33+side*21:54+side*21]):continue
                proposal=pose_palm_proposal(body[1],side)
                if proposal is None:continue
                try:hand=self.hand.infer(image,proposal)
                except cv2.error as error:
                    if 'resize' in str(error) or 'cvtColor' in str(error):continue
                    raise
                if hand is not None:points=assign_hands(points,[hand],width,height,fixed_side=side)
        return points


def extract_cpu_video(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot open video: {path}')
    points=[];extractor=CPUExtractor()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            points.append(extractor.process(frame))
    finally:cap.release()
    if not points:raise ValueError('Video contains no decodable frames')
    return np.stack(points)
