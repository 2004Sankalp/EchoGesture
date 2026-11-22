"""OpenHands-compatible XY features. XYZ retained for future trained adapters.

This baseline normalizes translation and scale, not arbitrary 3D viewpoint.
Changing its coordinates to 3D requires retraining, not silently replacing inputs.
"""
import numpy as np

JOINTS = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46,
          49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]


def pack_landmarks(result):
    points = np.zeros((75, 3), dtype=np.float32)
    for field, offset, count in [('pose_landmarks', 0, 33),
                                 ('left_hand_landmarks', 33, 21),
                                 ('right_hand_landmarks', 54, 21)]:
        group = getattr(result, field, None)
        if group:
            points[offset:offset + count] = [(p.x, p.y, p.z) for p in group.landmark]
    return points


def prepare(points):
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (75, 3) or not len(x):
        raise ValueError('Expected a nonempty [frames,75,3] landmark sequence')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite landmarks')
    x = x[:, JOINTS, :2].copy()
    center = ((x[:, 3] + x[:, 4]) / 2).mean(axis=0)
    distance = np.linalg.norm(x[:, 3] - x[:, 4], axis=1).mean()
    if distance > 0:
        x = (x - center) / distance
    # Match upstream CTV convention exactly, including missing-joint zeros.
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def extract_video(path):
    import os
    extractor=os.environ.get('ECHOGESTURE_EXTRACTOR','opencv')
    if extractor=='opencv':
        from .opencv_landmarks import extract_cpu_video
        return extract_cpu_video(path)
    if extractor!='mediapipe':raise ValueError('Extractor must be opencv or mediapipe')
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    points = []
    try:
        # No network calls. This model is bundled with the pinned MediaPipe wheel.
        with mp.solutions.holistic.Holistic(model_complexity=1,
                static_image_mode=False, min_detection_confidence=0.5) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                points.append(pack_landmarks(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    if not points:
        raise ValueError('Video contains no decodable frames')
    return np.stack(points)
