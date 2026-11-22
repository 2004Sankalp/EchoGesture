"""Disclosed software replay, not live camera/device validation.

Usage: python demo/render_walkthrough.py --archive /path/WLASL-full.zip
       --manifest /path/manifest-video-clean.json
Requires the runtime dependencies plus Pillow and imageio-ffmpeg.
"""
import argparse,io,json,sys,zipfile,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from echogesture.engine import EnsembleRecognizer
from echogesture.features3d import canonical_hand
from echogesture.conversation import Conversation
from prepare_public_poses import ArrayUnpickler
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();out=ROOT/'demo';engine=EnsembleRecognizer();rows=[r for r in json.load(open(a.manifest)) if r['split']=='train' and not r['clip_id'].startswith('citizen_')];archive=zipfile.ZipFile(a.archive);index={Path(n).stem:n for n in archive.namelist() if n.endswith('.pkl')};clips=[]
for word in ['BOOK','WATER','HELP']:
 candidates=[r for r in rows if r['label']==word]
 for r in candidates[:12]:
  points=np.asarray(ArrayUnpickler(io.BytesIO(archive.read(index[r['clip_id']]))).load()['keypoints'],np.float32);result=engine.classify(points)
  if result['prediction']==word:
   observed=[]
   for name,start in [('left',33),('right',54)]:
    shape,basis,valid=canonical_hand(points[:,start:start+21]);observed.append((int(valid.sum()),name,shape[valid.astype(bool)]))
   count,name,hand=max(observed,key=lambda v:v[0])
   if count<2:continue
   clips.append(dict(clip_id=r['clip_id'],word=word,points=points,result=result,hand=hand,visual_hand=name,visual_observed_frames=count));break
if len(clips)<3:raise ValueError('Need three disclosed training replay examples')
fontpath='/System/Library/Fonts/Supplemental/Arial.ttf'
def font(size):return ImageFont.truetype(fontpath,size)
F={s:font(s) for s in [16,18,20,22,24,28,32,40,48,58]};BG='#0B1728';MINT='#9EE9CF';WHITE='#F4F7FC';GREY='#AAB8CB';AMBER='#F3C579';fps=12;seconds=135;writer=imageio_ffmpeg.write_frames(str(out/'EchoGesture-Walkthrough.mp4'),(1280,720),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,macro_block_size=16,output_params=['-movflags','+faststart']);writer.send(None);conversation=Conversation();receipt=[];samples=[]
chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20],[5,9,13,17]]
def text(d,xy,value,size=24,color=WHITE):d.text(xy,value,font=F[size],fill=color)
def lines(d,x,y,values,size=24,color=GREY,gap=38):
 for i,v in enumerate(values):text(d,(x,y+i*gap),v,size,color)
def button(d,xy,label,active=False):
 x,y=xy;d.rounded_rectangle((x,y,x+164,y+48),radius=12,fill=MINT if active else '#21334B');text(d,(x+18,y+9),label,22,BG if active else WHITE)
last=-1
for frame in range(seconds*fps):
 t=frame/fps;im=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(im);text(d,(46,28),'EchoGesture',40,MINT);text(d,(46,84),'Local isolated ASL recognition. The signer reviews every word.',24);text(d,(46,669),'SOFTWARE REPLAY | public training poses | scripted confirmations | not a live phone test',18,AMBER)
 if t<18:
  text(d,(46,167),'A prediction is a suggestion.',48);text(d,(46,229),'You decide what it says.',48)
  lines(d,46,335,['Capture one isolated sign locally.','Review alternatives. Confirm, undo or try again.','Keep control of the caption and optional speech.'],28,gap=47)
  text(d,(46,548),'This walkthrough replays three disclosed training examples.',22,AMBER);text(d,(46,582),'It demonstrates software behavior, not benchmark accuracy.',22,AMBER)
 elif t<93:
  k=min(2,int((t-18)//25));local=(t-18)-25*k;c=clips[k]
  if k!=last:conversation.propose(c['result']);last=k;receipt.append({key:c[key] for key in ['clip_id','word','result','visual_hand','visual_observed_frames']})
  if 13<=local<13+1/fps:conversation.confirm(0)
  d.rounded_rectangle((46,149,737,611),radius=24,fill='#13263D');text(d,(76,174),'CANONICAL 3D HAND',22,MINT);text(d,(76,211),'Inferred depth; palm-relative coordinates',18,GREY)
  q=c['hand'][int((local%3)/3*len(c['hand']))].copy();angle=.45*math.sin(local*.45);rotation=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]]);q=q@rotation.T;xy=np.stack((q[:,0],-q[:,1]),1)*105+np.array([395,449]);
  for chain in chains:
   for left,right in zip(chain[:-1],chain[1:]):d.line((tuple(xy[left]),tuple(xy[right])),fill=MINT,width=5)
  for x,y in xy:d.ellipse((x-6,y-6,x+6,y+6),fill=WHITE)
  text(d,(770,155),'CAPTURING REPLAY' if local<6 else ('REVIEW A WORD' if local<13 else 'CONFIRMED'),22,MINT);text(d,(770,201),c['result']['prediction'].lower(),48)
  for j,word in enumerate(c['result']['top5']):text(d,(770,277+j*37),f'{j+1}   {word.lower().replace("_"," ")}',22,WHITE if j==0 else GREY)
  button(d,(770,488),'Confirm',6<=local<13);button(d,(956,488),'Try again')
  text(d,(770,564),'Caption: '+' '.join(conversation.caption).lower(),20,MINT);text(d,(76,569),'Training clip '+c['clip_id']+' | observed hand frames | display view rotates',16,GREY)
 elif t<113:
  text(d,(46,164),'Review is part of the design.',48);lines(d,46,269,['Candidates are proposals, not automatic speech.','Undo corrects a word. Clear removes the caption.','Typed replies support the return conversation.','Android Speak selects an installed offline English voice.'],28,gap=53);button(d,(46,524),'Undo',t<103);button(d,(236,524),'Clear',t>=103);text(d,(46,601),'Voice playback and handset behavior still need physical validation.',22,AMBER)
 else:
  text(d,(46,162),'What the evidence supports',48);lines(d,46,259,['308-channel inferred-XYZ geometry; INT8 classifier graphs.','All 2,000 provisional WLASL outputs retained.','Accuracy and deployment evidence are supplied separately.'],28,gap=48);lines(d,46,444,['70% top-1 remains unmet. Phone, power, robustness and thermal checks are pending.','Pose source: OpenHands WLASL, Zenodo 6674324; attribution in repository.','No original video frames, live phone recording or user study is represented here.'],20,AMBER,gap=36)
 d.rectangle((46,641,1234,646),fill='#243A52');d.rectangle((46,641,46+1188*t/seconds,646),fill=MINT)
 if frame in [5*fps,25*fps,55*fps,80*fps,103*fps,125*fps]:samples.append(im.resize((640,360)))
 writer.send(np.asarray(im))
writer.close();archive.close();sheet=Image.new('RGB',(1280,1080),BG)
for i,im in enumerate(samples):sheet.paste(im,((i%2)*640,(i//2)*360))
sheet.save(out/'walkthrough-contact-sheet.png');(out/'replay_receipt.json').write_text(json.dumps(dict(duration_seconds=seconds,scope='Disclosed training-pose software replay, not live camera or evaluation; missing hand frames omitted from visualization only, inference uses full clip',examples=receipt,model_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in engine.model_paths}),indent=2));print(out/'EchoGesture-Walkthrough.mp4',flush=True)
