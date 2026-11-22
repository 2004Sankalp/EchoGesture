"""Original WLASL first, then add the frozen Citizen training cohort."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
root=Path('work/unisign-study/features-mp');process=None
all_rows=[r for r in json.load(open('work/3d-data-citizen-v3/manifest-video-clean.json')) if r['split'] in ('train','val')];original=[r for r in all_rows if not r['clip_id'].startswith('citizen_')]
def stop(signum,frame):
 if process is not None and process.poll() is None:process.send_signal(signal.SIGINT)
 raise SystemExit(128+signum)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
while not (root/(original[-1]['clip_id']+'.record.json')).exists():time.sleep(3)
assert all((root/(r['clip_id']+'.record.json')).exists() for r in original)
(root/'manifest-stageA.json').write_text(json.dumps(original,indent=2));print(json.dumps({'stageA_ready':True,'train':13771,'val':3865,'classes':2000}),flush=True)
def run(name,channels,manifest,initialize=None):
 global process
 command=[sys.executable,'research/pose_transfer/train_pooled.py','--features',str(root),'--manifest',str(manifest),'--output','work/unisign-study/heads-mp/'+name,'--channels',str(channels),'--mode','noface','--epochs','100','--max-seconds','1800']
 if initialize:command.extend(['--initialize',initialize,'--lr','.0002','--epochs','70'])
 with open('work/unisign-study/train-'+name+'.log','w') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));print(json.dumps({'started':name,'pid':process.pid}),flush=True)
  try:
   if process.wait()!=0:raise RuntimeError('Training failed '+name)
  finally:
   if process.poll() is None:process.send_signal(signal.SIGINT)
for width in [128,256]:run('stageA'+str(width),width,root/'manifest-stageA.json')
while not (root/'coverage.json').exists():time.sleep(3)
for width in [128,256]:run('full'+str(width),width,root/'manifest.json','work/unisign-study/heads-mp/stageA'+str(width)+'/best.pt')
