"""Palm-centred 3D hand geometry plus separate position/orientation context.

MediaPipe normalised coordinates are inferred, not calibrated depth. Hand z is
wrist-relative; body z has another origin. Never combine them into global XYZ.
Canonical hand shape is invariant to a proper rigid rotation, translation and
positive scale of a nondegenerate hand cloud. The complete representation keeps
orientation context because orientation can change a sign's meaning.
"""
import numpy as np

FEATURE_VERSION = 'palm3d-context-v3'
SUPPORTED_VERSIONS = ('palm3d-context-v2', FEATURE_VERSION)
BODY = [0, 11, 12, 13, 14, 15, 16]
FEATURE_DIM = 308


def canonical_hand(points):
    """Return canonical [T,21,3], orientation [T,3,3], valid [T]."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite [T,21,3] hand points')
    centred = points - points[:, :1]
    lateral = centred[:, 5] - centred[:, 17]
    lateral_length = np.linalg.norm(lateral, axis=1)
    forward = centred[:, 9]
    longitudinal_length = np.linalg.norm(forward, axis=1)
    scale = (lateral_length + longitudinal_length) / 2
    x = lateral / np.maximum(lateral_length[:, None], 1e-7)
    forward = forward - (forward * x).sum(1, keepdims=True) * x
    forward_length = np.linalg.norm(forward, axis=1)
    y = forward / np.maximum(forward_length[:, None], 1e-7)
    z = np.cross(x, y)
    basis = np.stack((x, y, z), axis=-1)
    valid = (scale > 1e-6) & (lateral_length > 1e-6) & (forward_length > 0.05 * scale)
    canonical = np.matmul(centred, basis) / np.maximum(scale[:, None, None], 1e-7)
    canonical[~valid] = 0
    basis[~valid] = 0
    return canonical, basis, valid.astype(np.float32)


def describe(points, body_frame=True):
    """Variable-length [T,75,3] -> [T,308], retaining missingness explicitly."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (75, 3) or len(points) == 0:
        raise ValueError('Expected nonempty [T,75,3] landmarks')
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite landmarks')
    body = points[:, BODY].copy()
    present = np.any(body != 0, axis=-1)
    shoulders_valid = present[:, 1] & present[:, 2]
    centre = (body[:, 1] + body[:, 2]) / 2
    distances = np.linalg.norm(body[:, 1] - body[:, 2], axis=1)
    good = distances[shoulders_valid & (distances > 1e-5)]
    scale = float(np.median(good)) if len(good) else 1.0
    rotation=np.tile(np.eye(3,dtype=np.float32),(len(points),1,1))
    if body_frame:
        shoulder_xy=points[:,11,:2]-points[:,12,:2]
        xy_length=np.linalg.norm(shoulder_xy,axis=1)
        valid_rotation=shoulders_valid & (xy_length>1e-5)
        ex=shoulder_xy/np.maximum(xy_length[:,None],1e-5)
        rotation[valid_rotation,0,:2]=ex[valid_rotation]
        rotation[valid_rotation,1,0]=-ex[valid_rotation,1]
        rotation[valid_rotation,1,1]=ex[valid_rotation,0]
    body = (body - centre[:, None]) / scale
    body=np.matmul(body,rotation.transpose(0,2,1))
    body[~present] = 0
    body[~shoulders_valid] = 0
    # Body wrists supply body-relative movement. Hand z is never used here.
    wrist_velocity = np.zeros((len(points), 2, 3), dtype=np.float32)
    if len(points) > 1:
        wrist_velocity[1:] = body[1:, 5:7] - body[:-1, 5:7]
        consecutive = present[1:, 5:7] & present[:-1, 5:7] & shoulders_valid[1:, None] & shoulders_valid[:-1, None]
        wrist_velocity[1:][~consecutive] = 0
    parts = [body.reshape(len(points), -1), present.astype(np.float32)]
    for start in (33, 54):
        shape, basis, valid = canonical_hand(points[:, start:start+21])
        basis=np.matmul(rotation,basis)
        parts.extend((shape.reshape(len(points), -1), basis.reshape(len(points), -1), valid[:, None]))
    parts.append(wrist_velocity.reshape(len(points), -1))
    # Preserve local pose/orientation information alongside canonical shape.
    # XY is body-relative; hand Z stays wrist-relative, never global body Z.
    for start in (33, 54):
        hand=points[:,start:start+21]
        valid=np.any(hand!=0,axis=(1,2)) & shoulders_valid
        raw=hand.copy()
        raw[:,:,:2]=(hand[:,:,:2]-centre[:,None,:2])/scale
        raw[:,:,2]=(hand[:,:,2]-hand[:,:1,2])/scale
        raw=np.matmul(raw,rotation.transpose(0,2,1))
        raw[~valid]=0
        size=np.linalg.norm(hand[:,9]-hand[:,0],axis=1)/scale
        size[~valid]=0
        parts.extend((raw.reshape(len(points),-1),size[:,None]))
    result = np.concatenate(parts, axis=1)
    assert result.shape[1] == FEATURE_DIM
    return np.clip(result, -8, 8).astype(np.float32)


def prepare3d(points, frames=48, version=FEATURE_VERSION):
    """Deterministic nearest-frame resampling, no interpolation of missing hands."""
    if version not in SUPPORTED_VERSIONS:raise ValueError('Unsupported 3D feature version')
    features = describe(points,body_frame=version==FEATURE_VERSION)
    indices = np.rint(np.linspace(0, len(features)-1, frames)).astype(int)
    return np.ascontiguousarray(features[indices].T, dtype=np.float32)
