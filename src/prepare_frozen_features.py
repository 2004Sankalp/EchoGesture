"""Recreate exact final feature order from the shipped metadata-only manifest.

Original videos/pose archives are not redistributed. This command never trains
on test data: it prepares immutable features; train_3d.py loads train/val only.
"""
import argparse,gzip,json,io,hashlib,os,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from prepare_public_poses import entries,ArrayUnpickler
from echogesture.features3d import prepare3d

def atomic_tensor(path,value):
    temporary=path.with_suffix('.tmp.npy');np.save(temporary,value);os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,help='Verified WLASL-full.zip from Zenodo6674324');p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--skip-citizen',action='store_true');a=p.parse_args()
    package=Path(__file__).resolve().parents[1];protocol=package/'resources/protocol'
    frozen=gzip.decompress((protocol/'training_manifest_video_clean.json.gz').read_bytes());rows=json.loads(frozen);coverage=gzip.decompress((protocol/'training_coverage_video_clean.json.gz').read_bytes())
    labels=json.loads((package/'models/labels.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with open(a.archive,'rb') as archive_file:
        if hashlib.file_digest(archive_file,'md5').hexdigest()!='c45b690a5340da71d9a75db19e0cc572':raise ValueError('WLASL archive differs from the verified publisher MD5')
    for row in rows:
        if Path(row['features']).name!=row['features'] or row['label']!=labels[row['class_id']]:raise ValueError('Invalid frozen row')
    wanted={r['clip_id']:r for r in rows if not r['clip_id'].startswith('citizen_')};done=set()
    for name,raw in entries(a.archive):
        clip=Path(name).stem
        if not name.endswith('.pkl') or clip not in wanted:continue
        path=out/wanted[clip]['features']
        if not path.exists():
            points=ArrayUnpickler(io.BytesIO(raw)).load()['keypoints'];atomic_tensor(path,prepare3d(points,48,'palm3d-context-v3').astype(np.float16))
        done.add(clip)
    if done!=wanted.keys():raise ValueError('Incomplete WLASL archive')
    if a.skip_citizen:return
    import requests
    from pose_format import Pose
    from echogesture.citizen import to_mediapipe75
    local=threading.local();stop=threading.Event()
    def process(row):
        path=out/row['features']
        if path.exists():
            value=np.load(path,allow_pickle=False)
            if value.shape!=(308,48) or hashlib.sha256(value.tobytes()).hexdigest()!=row['feature_sha256']:raise ValueError('Existing feature mismatch')
            return
        url=row['source_url']
        if not url.startswith('https://huggingface.co/datasets/SorensenAI/asl-citizen-poses/resolve/'):raise ValueError('Unexpected data origin')
        if not hasattr(local,'session'):local.session=requests.Session()
        for attempt in range(6):
            if stop.is_set():raise InterruptedError('Stopped')
            try:
                response=local.session.get(url,timeout=(15,45));response.raise_for_status();raw=response.content
                if hashlib.sha256(raw).hexdigest()!=row['source_sha256']:raise ValueError('Source SHA256 mismatch')
                break
            except requests.RequestException as error:
                if isinstance(error,requests.HTTPError) and error.response is not None and error.response.status_code in (401,403,404):raise
                if attempt==5:raise
                if stop.wait(min(45,5*(attempt+1))):raise InterruptedError('Stopped')
        value=prepare3d(to_mediapipe75(Pose.read(raw)),48,'palm3d-context-v3').astype(np.float16)
        if hashlib.sha256(value.tobytes()).hexdigest()!=row['feature_sha256']:raise ValueError('Recreated feature differs; check pinned NumPy/pose-format versions')
        atomic_tensor(path,value)
    citizen=[r for r in rows if r['clip_id'].startswith('citizen_')];pool=ThreadPoolExecutor(max_workers=a.workers)
    try:
        futures=[pool.submit(process,r) for r in citizen]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%100==0:print(json.dumps({'citizen_complete':i,'total':len(citizen)}),flush=True)
    except KeyboardInterrupt:stop.set();raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    # Exact original bytes preserve the dataset fingerprint in checkpoints.
    (out/'manifest-video-clean.json').write_bytes(frozen);(out/'manifest.json').write_bytes(frozen)
    (out/'coverage-video-clean.json').write_bytes(coverage);(out/'coverage.json').write_bytes(coverage)
    (out/'labels.json').write_text(json.dumps(labels,indent=2))
    print(json.dumps({'train':30730,'val':3865,'test':2878,'classes':2000,'manifest_sha256':hashlib.sha256(frozen).hexdigest(),'no_training_performed':True}),flush=True)
if __name__=='__main__':main()
