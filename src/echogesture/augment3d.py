"""Training-only coordinate augmentation for normalised camera geometry."""
import torch


def stretch_vertical(features, factors):
    """Stretch body-relative Y, recomputing 3D palm frames and hand scale.

    This approximates a change in normalised image aspect ratio. It preserves
    missing detections and does not swap anatomical hands or change labels.
    """
    x=features.clone();batch,_,frames=x.shape
    factors=factors.to(dtype=x.dtype,device=x.device).reshape(batch,1,1)
    body=x[:,:21].reshape(batch,7,3,frames)
    body[:,:,1]*=factors
    velocity=x[:,174:180].reshape(batch,2,3,frames)
    velocity[:,:,1]*=factors
    for raw_start,shape_start,basis_start,valid_index in [(180,28,91,100),(244,101,164,173)]:
        hand=x[:,raw_start:raw_start+63].transpose(1,2).reshape(batch,frames,21,3).clone()
        hand[:,:,:,1]*=factors
        local=hand-hand[:,:,:1]
        lateral=local[:,:,5]-local[:,:,17]
        forward=local[:,:,9]
        lateral_length=torch.linalg.vector_norm(lateral,dim=-1)
        longitudinal_length=torch.linalg.vector_norm(forward,dim=-1)
        scale=(lateral_length+longitudinal_length)/2
        axis_x=lateral/lateral_length.clamp_min(1e-7).unsqueeze(-1)
        forward=forward-(forward*axis_x).sum(-1,keepdim=True)*axis_x
        forward_length=torch.linalg.vector_norm(forward,dim=-1)
        axis_y=forward/forward_length.clamp_min(1e-7).unsqueeze(-1)
        axis_z=torch.linalg.cross(axis_x,axis_y,dim=-1)
        basis=torch.stack((axis_x,axis_y,axis_z),dim=-1)
        canonical=torch.matmul(local,basis)/scale.clamp_min(1e-7)[:,:,None,None]
        present=(hand.abs().sum((-1,-2))>0)&(x[:,valid_index]>0)
        geometric=(lateral_length>1e-6)&(forward_length>0.05*scale)&(scale>1e-6)
        valid=present&geometric
        old_shape=x[:,shape_start:shape_start+63].transpose(1,2).reshape(batch,frames,21,3)
        old_basis=x[:,basis_start:basis_start+9].transpose(1,2).reshape(batch,frames,3,3)
        # If body coordinates were unavailable, leave the independently detected
        # canonical hand untouched instead of inventing body-relative geometry.
        canonical=torch.where(present[:,:,None,None],canonical,old_shape)
        basis=torch.where(present[:,:,None,None],basis,old_basis)
        canonical=torch.where((~present|valid)[:,:,None,None],canonical,torch.zeros_like(canonical))
        basis=torch.where((~present|valid)[:,:,None,None],basis,torch.zeros_like(basis))
        x[:,raw_start:raw_start+63]=hand.reshape(batch,frames,63).transpose(1,2)
        x[:,raw_start+63]=torch.linalg.vector_norm(local[:,:,9],dim=-1)
        x[:,shape_start:shape_start+63]=canonical.reshape(batch,frames,63).transpose(1,2)
        x[:,basis_start:basis_start+9]=basis.reshape(batch,frames,9).transpose(1,2)
        x[:,valid_index]=torch.where(present,valid.to(x.dtype),x[:,valid_index])
    return x.clamp(-8,8)
