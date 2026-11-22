import sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_video_identities import audit

def test_byte_duplicates_removed_from_earlier_splits_and_test_preserved(tmp_path):
    items=[('A','train',0,b'one'),('B','val',0,b'one'),('C','test',1,b'two'),('D','val',1,b'two'),('E','train',0,b'three'),('F','train',1,b'four')]
    path=tmp_path/'videos.zip';rows=[]
    with zipfile.ZipFile(path,'w') as z:
        for clip,split,label,data in items:
            z.writestr(split+'/'+clip+'-wlasl-word.mp4',data)
            rows.append(dict(clip_id=clip,split=split,class_id=label))
    clean,report=audit(path,rows)
    assert report['excluded_ids']==['A','D']
    assert [r['clip_id'] for r in clean]==['B','C','E','F']
    assert report['training_classes_after']==report['training_classes_before']==2
    assert [r for r in clean if r['split']=='test']==[r for r in rows if r['split']=='test']
