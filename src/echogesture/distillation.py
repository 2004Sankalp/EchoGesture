"""Align teacher logits to training IDs only, rejecting held-out inputs."""
import numpy as np


def load_training_teacher(path,rows,labels):
    with np.load(path,allow_pickle=False) as archive:
        ids=archive['clip_ids'].tolist()
        truth=archive['truth']
        logits=archive['logits']
        if archive['labels'].tolist()!=labels:raise ValueError('Teacher vocabulary/order differs')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate teacher IDs')
        if logits.shape!=(len(ids),len(labels)) or truth.shape!=(len(ids),):raise ValueError('Teacher shapes differ')
        if not np.isfinite(logits).all():raise ValueError('Nonfinite teacher logits')
        lookup={r['clip_id']:i for i,r in enumerate(rows)}
        if len(lookup)!=len(rows) or any(r['split']!='train' for r in rows):raise ValueError('Expected unique training rows')
        if not set(ids)<=lookup.keys():raise ValueError('Teacher contains IDs outside student training split')
        aligned=np.zeros((len(rows),len(labels)),dtype=np.float16)
        valid=np.zeros(len(rows),dtype=np.float32)
        for source,clip_id in enumerate(ids):
            target=lookup[clip_id]
            if int(truth[source])!=rows[target]['class_id']:raise ValueError('Teacher truth differs')
            aligned[target]=logits[source];valid[target]=1
    return aligned,valid
