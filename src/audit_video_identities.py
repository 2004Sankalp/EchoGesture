"""Verify exact cross-split video duplicates, retaining every original test row.

Only byte identity is examined; no test image decoding or model predictions.
CRC32/size screen candidates, and SHA256 verifies identities within candidates.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
from collections import defaultdict,Counter

def audit(archive,rows):
    groups=defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        info={}
        for item in z.infolist():
            if not item.filename.endswith('.mp4'):continue
            clip=Path(item.filename).stem.split('-wlasl-',1)[0]
            if clip in info:raise ValueError('Ambiguous video ID '+clip)
            info[clip]=item
        if len({r['clip_id'] for r in rows})!=len(rows):raise ValueError('Duplicate manifest IDs')
        for row in rows:
            item=info[row['clip_id']];groups[(item.CRC,item.file_size)].append(row)
        confirmed=[];exclude=set()
        for group in groups.values():
            if len({r['split'] for r in group})<2:continue
            by_hash=defaultdict(list)
            for row in group:by_hash[hashlib.sha256(z.read(info[row['clip_id']])).hexdigest()].append(row)
            for digest,duplicates in by_hash.items():
                splits={r['split'] for r in duplicates}
                if len(splits)<2:continue
                confirmed.append(dict(sha256=digest,rows=duplicates))
                for row in duplicates:
                    if row['split']=='train' or (row['split']=='val' and 'test' in splits):exclude.add(row['clip_id'])
    clean=[r for r in rows if r['clip_id'] not in exclude]
    if [r for r in clean if r['split']=='test']!=[r for r in rows if r['split']=='test']:raise AssertionError('Test set changed')
    return clean,dict(scope='Video SHA256 identity only; no test decoding/predictions',confirmed=confirmed,
        excluded_ids=sorted(exclude),counts=dict(Counter(r['split'] for r in clean)),
        original_counts=dict(Counter(r['split'] for r in rows)),
        training_classes_before=len({r['class_id'] for r in rows if r['split']=='train'}),
        training_classes_after=len({r['class_id'] for r in clean if r['split']=='train'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=json.loads(Path(a.manifest).read_text());clean,report=audit(a.archive,rows);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(clean,indent=2));(out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='confirmed'},indent=2))
if __name__=='__main__':main()
