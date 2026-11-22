"""Wait for the fixed encoder cache, apply video-identity hygiene, train heads."""
import sys,signal,time,json,subprocess,os
from pathlib import Path
from collections import Counter
sys.path.insert(0,'src')
from cache_training_tensors import build_cache
processes=[];streams=[]
def stop(signum,frame):
 for p in processes:
  if p.poll() is None:p.send_signal(signal.SIGINT)
 raise SystemExit(128+signum)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
root=Path('work/appearance-views-data');spatial=root/'spatial'
while True:
 try:
  coverage=json.loads((root/'coverage.json').read_text());detail=json.loads((spatial/'coverage.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2)
clean=[r for r in json.load(open('work/manifest-wlasl-video-clean.json')) if r['split'] in ('train','val')]
assert len({r['class_id'] for r in clean if r['split']=='train'})==2000
for directory,info in [(root,coverage),(spatial,detail)]:
 assert all((directory/r['features']).exists() for r in clean)
 (directory/'manifest.json').write_text(json.dumps(clean,indent=2));info.update(counts=dict(Counter(r['split'] for r in clean)),protocol='Video-SHA and pose-hash clean; train-only original WLASL; independent ImageNet encoder; fresh ASL head',identity_audit='work/appearance/video-identity-audit.json')
 (directory/'coverage.json').write_text(json.dumps(info,indent=2))
 print(json.dumps({'building_shared_cache':str(directory),'counts':info['counts']}),flush=True)
 build_cache(directory,directory/'manifest.json',directory/'shared')
try:
 for name,directory,architecture,seed in [('audited-rgb-global128',root,'motion',42),('audited-rgb-spatial128',spatial,'appearance',73)]:
  stream=open('work/train-'+name+'.log','w');streams.append(stream)
  command=[sys.executable,'src/train_3d.py','--data',str(directory),'--tensor-cache',str(directory/'shared'),'--output','work/experiments/'+name,'--architecture',architecture,'--classifier','cosine','--channels','128','--epochs','90','--patience','15','--threads','1','--batch','64','--lr','.0015','--balanced','--max-seconds','10800','--seed',str(seed)]
  processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')));print(json.dumps({'started':name,'pid':processes[-1].pid,'fresh_ASL_head':True}),flush=True)
 for p in processes:p.wait()
finally:
 for p in processes:
  if p.poll() is None:p.send_signal(signal.SIGINT)
 for stream in streams:stream.close()
