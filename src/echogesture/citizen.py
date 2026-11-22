"""Convert original pose-format Holistic coordinates to normalized 75-point input."""
import numpy as np


def to_mediapipe75(pose):
    if pose.body.data.ndim!=4 or pose.body.data.shape[1]!=1:
        raise ValueError('Expected exactly one signer')
    components={};offset=0
    for component in pose.header.components:
        components[component.name]=(offset,len(component.points));offset+=len(component.points)
    selected=[]
    for name,count in [('POSE_LANDMARKS',33),('LEFT_HAND_LANDMARKS',21),('RIGHT_HAND_LANDMARKS',21)]:
        start,size=components[name]
        if size!=count:raise ValueError('Unexpected landmark topology')
        selected.extend(range(start,start+size))
    points=np.asarray(pose.body.data.filled(0)[:,0,selected,:],dtype=np.float32).copy()
    if points.shape[-1]!=3:raise ValueError('Expected XYZ coordinates')
    points[pose.body.confidence[:,0,selected]<=0]=0
    width,height=pose.header.dimensions.width,pose.header.dimensions.height
    if width<=0 or height<=0:raise ValueError('Invalid image dimensions')
    # pose-format stores MediaPipe X/Y in pixels, but keeps Z unscaled.
    points[:,:,0]/=width;points[:,:,1]/=height
    if not np.isfinite(points).all():raise ValueError('Nonfinite pose coordinates')
    return points
