import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cache_training_tensors import build_cache


def test_shared_cache_preserves_alignment_and_never_reads_test(tmp_path):
    a=np.arange(15,dtype=np.float16).reshape(3,5);b=a+1
    np.save(tmp_path/'a.npy',a);np.save(tmp_path/'b.npy',b)
    rows=[{'clip_id':'a','features':'a.npy','split':'train','class_id':1},
          {'clip_id':'b','features':'b.npy','split':'val','class_id':0},
          {'clip_id':'heldout','features':'not-present.npy','split':'test','class_id':0}]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(rows))
    info=build_cache(tmp_path,manifest,tmp_path/'shared')
    np.testing.assert_array_equal(np.load(tmp_path/'shared/train.npy')[0],a)
    np.testing.assert_array_equal(np.load(tmp_path/'shared/val.npy')[0],b)
    np.testing.assert_array_equal(np.load(tmp_path/'shared/train_labels.npy'),[1])
    assert set(info['splits'])=={'train','val'}
