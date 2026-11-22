"""Create shared, disk-backed train/val arrays without reading test tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def build_cache(data,manifest_path,output):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=Path(manifest_path).read_bytes();manifest=json.loads(raw)
    info={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'splits':{}}
    receipt=output/'cache.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['manifest_sha256']!=info['manifest_sha256']:
            raise ValueError('Use a fresh cache directory for a different manifest')
        for split,entry in previous['splits'].items():
            tensor=np.load(output/(split+'.npy'),mmap_mode='r',allow_pickle=False)
            if list(tensor.shape)!=entry['shape']:raise ValueError('Existing cache shape differs')
        return previous
    for split in ('train','val'):
        rows=[r for r in manifest if r['split']==split]
        if not rows:raise ValueError(f'Missing {split} rows')
        sample=np.load(data/rows[0]['features'],allow_pickle=False)
        if sample.dtype!=np.float16 or sample.ndim!=2:raise ValueError('Expected stored float16 feature matrices')
        tensor=np.lib.format.open_memmap(output/(split+'.npy'),mode='w+',dtype=np.float16,shape=(len(rows),*sample.shape))
        for index,row in enumerate(rows):
            value=np.load(data/row['features'],allow_pickle=False)
            if value.shape!=sample.shape or value.dtype!=np.float16 or not np.isfinite(value).all():
                raise ValueError(f'Incompatible feature tensor: {row["clip_id"]}')
            tensor[index]=value
        tensor.flush()
        np.save(output/(split+'_labels.npy'),np.array([r['class_id'] for r in rows],dtype=np.int64))
        info['splits'][split]={'clips':len(rows),'shape':list(tensor.shape),'dtype':'float16'}
        del tensor
    temporary=output/'cache.json.tmp';temporary.write_text(json.dumps(info,indent=2));temporary.replace(receipt)
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build_cache(args.data,args.manifest,args.output)))
