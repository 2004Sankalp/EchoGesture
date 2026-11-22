"""Fresh ASL training under the stronger video-identity audit; pause-safe."""
import time,json,signal,subprocess,sys,os
from pathlib import Path
root=Path('work/3d-data-citizen-v3');processes=[];streams=[]
def stop(signum,frame):
 for p in processes:
  if p.poll() is None:p.send_signal(signal.SIGINT)
 raise SystemExit(128+signum)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
while not (root/'shared-video-clean/cache.json').exists():time.sleep(2)
try:
 for name,architecture,seed in [('audited-anatomy128','anatomy',42),('audited-kinematic128','anatomy_kinematic',73)]:
  log=open('work/train3d-'+name+'.log','a');streams.append(log)
  command=[sys.executable,'src/train_3d.py','--data',str(root),'--manifest',str(root/'manifest-video-clean.json'),'--coverage',str(root/'coverage-video-clean.json'),'--tensor-cache',str(root/'shared-video-clean'),'--output','work/experiments/'+name,'--architecture',architecture,'--classifier','cosine','--channels','128','--epochs','100','--patience','18','--threads','1','--batch','64','--lr','.0015','--balanced','--aspect-augment','--max-seconds','14400','--seed',str(seed)]
  resume=Path('work/experiments')/name/'last.pt'
  if resume.exists():command.extend(['--resume',str(resume)])
  environment=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
  processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment));print(json.dumps({'name':name,'pid':processes[-1].pid,'fresh_random_ASL_weights':True,'train':30730,'val':3865,'classes':2000}),flush=True)
 for p in processes:p.wait()
finally:
 for p in processes:
  if p.poll() is None:p.send_signal(signal.SIGINT)
 for stream in streams:stream.close()
