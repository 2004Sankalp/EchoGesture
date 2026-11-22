"""Cache 3D features preserving the upstream train/val/test partitions."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from prepare_public_poses import ArrayUnpickler, entries
from echogesture.features3d import prepare3d, FEATURE_VERSION, SUPPORTED_VERSIONS


def clean_splits(rows, hashes):
    """Keep test intact; exclude train/val identities seen in later splits."""
    test = {hashes[r['clip_id']] for r in rows if r['split']=='test'}
    held = {hashes[r['clip_id']] for r in rows if r['split'] in ('val','test')}
    kept=[];excluded=[]
    for row in rows:
        hash_value=hashes[row['clip_id']]
        if (row['split']=='train' and hash_value in held) or (row['split']=='val' and hash_value in test):
            excluded.append(row)
        else:kept.append(row)
    return kept,excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--feature-version', choices=SUPPORTED_VERSIONS, default=FEATURE_VERSION)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'coverage.json').exists():
        previous=json.loads((out/'coverage.json').read_text())
        if previous.get('feature_version')!=args.feature_version or previous.get('frames')!=args.frames:
            raise ValueError('Cache feature version/length differs. Choose a new output directory.')
    data = json.loads(Path(args.metadata).read_text())
    labels = sorted(e['gloss'].upper() for e in data)
    mapping = {label:i for i,label in enumerate(labels)}
    clips = {i['video_id']:dict(clip_id=i['video_id'], split=i['split'], label=e['gloss'].upper(),
        signer_id=i.get('signer_id'), class_id=mapping[e['gloss'].upper()]) for e in data for i in e['instances']}
    collected=[]; hashes={}; rejected=[]
    for name, raw in entries(args.archive):
        clip_id = Path(name).stem
        if not name.endswith('.pkl') or clip_id not in clips: continue
        row=dict(clips[clip_id]); path=out/(clip_id+'.npy')
        try:
            if not path.exists():
                points=ArrayUnpickler(io.BytesIO(raw)).load()['keypoints']
                features=prepare3d(points,args.frames,args.feature_version)
                np.save(path,features.astype(np.float16))
            hashes[clip_id]=hashlib.sha256(raw).hexdigest()
            row['features']=path.name;collected.append(row)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(dict(clip_id=clip_id,error=str(error)))
    official=list(collected)
    collected,excluded=clean_splits(official,hashes)
    identities={s:{r['signer_id'] for r in collected if r['split']==s} for s in ('train','val','test')}
    counts=Counter(r['split'] for r in collected)
    report=dict(feature_version=args.feature_version, frames=args.frames, classes=len(labels),
        counts=dict(counts), metadata_counts=dict(Counter(i['split'] for i in clips.values())),
        class_coverage={s:len({r['label'] for r in collected if r['split']==s}) for s in identities},
        signer_overlap_train_val=len(identities['train'] & identities['val']),
        signer_overlap_train_test=len(identities['train'] & identities['test']),
        signer_independent=False, source='OpenHands WLASL public inferred-XYZ poses',
        raw_clip_sha256=hashes, rejected=rejected,
        original_counts=dict(Counter(r['split'] for r in official)),
        exact_duplicate_exclusions=excluded,
        split_policy='Test intact. Train excludes any raw-pose SHA256 in val/test. Val excludes any raw-pose SHA256 in test. Signer overlap remains.')
    (out/'official_manifest.json').write_text(json.dumps(official,indent=2))
    (out/'labels.json').write_text(json.dumps(labels,indent=2));(out/'manifest.json').write_text(json.dumps(collected,indent=2));(out/'coverage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_clip_sha256','rejected','exact_duplicate_exclusions')},indent=2),flush=True)


if __name__=='__main__':main()
