"""NumPy geometry transforms shared by CPU deployment and evaluation."""
import numpy as np


def stretch_vertical(features, factor):
    """Recompute palm geometry after stretching normalized vertical coordinates."""
    x = np.asarray(features, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[0] != 308 or not np.isfinite(factor) or factor <= 0:
        raise ValueError('Expected 308 x frames features and a positive finite factor')
    frames = x.shape[1]
    x[:21].reshape(7, 3, frames)[:, 1] *= factor
    x[174:180].reshape(2, 3, frames)[:, 1] *= factor
    for raw, shape, orientation, validity in [(180, 28, 91, 100), (244, 101, 164, 173)]:
        hand = x[raw:raw+63].T.reshape(frames, 21, 3).copy()
        hand[:, :, 1] *= factor
        local = hand-hand[:, :1]
        lateral = local[:, 5]-local[:, 17]
        forward = local[:, 9].copy()
        lateral_length = np.linalg.norm(lateral, axis=-1)
        scale = (lateral_length+np.linalg.norm(forward, axis=-1))/2
        axis_x = lateral/np.maximum(lateral_length, 1e-7)[:, None]
        forward -= (forward*axis_x).sum(-1, keepdims=True)*axis_x
        forward_length = np.linalg.norm(forward, axis=-1)
        axis_y = forward/np.maximum(forward_length, 1e-7)[:, None]
        basis = np.stack((axis_x, axis_y, np.cross(axis_x, axis_y)), axis=-1)
        canonical = np.matmul(local, basis)/np.maximum(scale, 1e-7)[:, None, None]
        present = (np.abs(hand).sum((1, 2)) > 0) & (x[validity] > 0)
        valid = present & (lateral_length > 1e-6) & (forward_length > .05*scale) & (scale > 1e-6)
        canonical = np.where(present[:, None, None], canonical, x[shape:shape+63].T.reshape(frames, 21, 3))
        basis = np.where(present[:, None, None], basis, x[orientation:orientation+9].T.reshape(frames, 3, 3))
        canonical = np.where((~present | valid)[:, None, None], canonical, 0)
        basis = np.where((~present | valid)[:, None, None], basis, 0)
        x[raw:raw+63] = hand.reshape(frames, 63).T
        x[raw+63] = np.linalg.norm(local[:, 9], axis=-1)
        x[shape:shape+63] = canonical.reshape(frames, 63).T
        x[orientation:orientation+9] = basis.reshape(frames, 9).T
        x[validity] = np.where(present, valid, x[validity])
    return np.clip(x, -8, 8)
