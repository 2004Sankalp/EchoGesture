"""Training-only ASLLVD addition; fixed WLASL validation and test retained."""
import sys,io,json,csv,hashlib,os,argparse
from pathlib import Path
import numpy as np

from prepare_public_poses import entries,ArrayUnpickler
from echogesture.features3d import prepare3d
parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--output',required=True);parser.add_argument('--wlasl-archive',required=True);parser.add_argument('--asllvd-archive',required=True);parser.add_argument('--metadata',required=True);args=parser.parse_args()
base=Path(args.base);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
if json.loads((base/'coverage.json').read_text())['feature_version']!='palm3d-context-v3':raise ValueError('Extra cache requires v3 base features')
with open(args.asllvd_archive,'rb') as stream: archive_md5=hashlib.file_digest(stream,'md5').hexdigest()
if archive_md5!='cd73156728bbc3ad1d016a5decc5c6ec':raise ValueError('ASLLVD archive checksum differs')
rows=json.loads((base/'manifest.json').read_text());labels=json.loads((base/'labels.json').read_text());mapping={v:i for i,v in enumerate(labels)}
for row in rows:
 target=out/row['features']
 if not target.exists():os.link(base/row['features'],target)
held={r['clip_id'] for r in rows if r['split'] in ['val','test']};fingerprints=set()
for name,raw in entries(args.wlasl_archive):
 if Path(name).stem in held and name.endswith('.pkl'):
  p=ArrayUnpickler(io.BytesIO(raw)).load()['keypoints'];fingerprints.add(hashlib.sha256(np.asarray(p,dtype=np.float32).tobytes()).hexdigest())
meta=Path(args.metadata)
normal={r['actual_gloss']:r['normalized_gloss'] for r in csv.DictReader((meta/'normalized_glosses.csv').open())}
training=json.loads((meta/'train_labels.json').read_text());extra=[];excluded=[]
for name,raw in entries(args.asllvd_archive):
 key=Path(name).stem
 if not name.endswith('.pkl') or key not in training:continue
 label=normal.get(training[key]['label'])
 if label not in mapping:continue
 points=ArrayUnpickler(io.BytesIO(raw)).load()['keypoints'];sha=hashlib.sha256(np.asarray(points,dtype=np.float32).tobytes()).hexdigest()
 if sha in fingerprints:excluded.append(key);continue
 clip='asllvd_'+key;feature=clip+'.npy';np.save(out/feature,prepare3d(points,48,'palm3d-context-v3').astype(np.float16));extra.append(dict(clip_id=clip,features=feature,split='train',label=label,class_id=mapping[label],source='ASLLVD original training split',raw_pose_sha256=sha))
coverage=json.loads((base/'coverage.json').read_text());coverage['external_training']={'source':'OpenHands ASLLVD-Skeleton.zip / original ASLLVD train_labels.json only','archive_md5':'cd73156728bbc3ad1d016a5decc5c6ec','label_policy':'Exact normalized_gloss equality only; no guessed synonyms; original ASLLVD test unused','clips':len(extra),'classes':len({r['label'] for r in extra}),'heldout_exact_pose_hash_excluded':excluded,'limitations':'Exact array hash exclusion only, does not establish no near-duplicate video content.'};coverage['counts']['train']+=len(extra)
(out/'coverage.json').write_text(json.dumps(coverage,indent=2));(out/'manifest.json').write_text(json.dumps(rows+extra,indent=2));(out/'labels.json').write_text(json.dumps(labels,indent=2));print(coverage['external_training'])
