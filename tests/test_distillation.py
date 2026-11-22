import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echogesture.distillation import load_training_teacher

def test_teacher_aligns_by_training_id_and_keeps_unavailable_rows_masked(tmp_path):
    rows=[dict(clip_id='A',split='train',class_id=0),dict(clip_id='B',split='train',class_id=1)]
    p=tmp_path/'teacher.npz'
    np.savez(p,clip_ids=['B'],truth=[1],logits=np.array([[2.,4.]],np.float16),labels=['ONE','TWO'])
    logits,valid=load_training_teacher(p,rows,['ONE','TWO'])
    np.testing.assert_array_equal(logits,[[0,0],[2,4]])
    np.testing.assert_array_equal(valid,[0,1])
    np.savez(p,clip_ids=['HELDOUT'],truth=[1],logits=np.array([[2.,4.]],np.float16),labels=['ONE','TWO'])
    with pytest.raises(ValueError,match='outside student training'):load_training_teacher(p,rows,['ONE','TWO'])

def test_teacher_rejects_wrong_labels_and_duplicate_ids(tmp_path):
    rows=[dict(clip_id='A',split='train',class_id=0)];p=tmp_path/'teacher.npz'
    np.savez(p,clip_ids=['A'],truth=[1],logits=np.array([[2.,4.]],np.float16),labels=['ONE','TWO'])
    with pytest.raises(ValueError,match='truth differs'):load_training_teacher(p,rows,['ONE','TWO'])
    np.savez(p,clip_ids=['A','A'],truth=[0,0],logits=np.zeros((2,2),np.float16),labels=['ONE','TWO'])
    with pytest.raises(ValueError,match='Duplicate'):load_training_teacher(p,rows,['ONE','TWO'])
