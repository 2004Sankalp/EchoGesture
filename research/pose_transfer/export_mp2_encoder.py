import json,time,hashlib
from pathlib import Path
import torch,numpy as np,onnx,onnxruntime as ort
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType
from encoder import PoseEncoder
from mp_parts import from_features
root=Path('work/unisign-study');torch.set_num_threads(1);state=torch.load(root/'openasl_pose_only_slt.pth',map_location='cpu',weights_only=True,mmap=True)['model'];model=PoseEncoder(state,include_face=False).eval();inputs=tuple(torch.randn(1,32,n,3) for n in [9,21,21]);names=['body','left','right'];fp=root/'encoder-mp.fp32.onnx';target=root/'encoder-mp2.int8.onnx'
torch.onnx.export(model,inputs,str(fp),input_names=names,output_names=['pose_features','tail_inputs'],opset_version=17,dynamo=False,dynamic_axes={**{n:{0:'batch'} for n in names},'pose_features':{0:'batch'},'tail_inputs':{0:'batch'}})
onnx.checker.check_model(str(fp));o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;session=ort.InferenceSession(str(fp),o)
with torch.inference_mode():expected=[x.numpy() for x in model(*inputs)]
actual=session.run(None,dict(zip(names,[x.numpy() for x in inputs])))
for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=.001,rtol=.001)
data=Path('work/3d-data-citizen-v3');rows=[r for r in json.load(open(data/'manifest-video-clean.json')) if r['split']=='train'];ordered=sorted(rows,key=lambda r:hashlib.sha256(r['clip_id'].encode()).hexdigest());first={}
for r in ordered:first.setdefault(r['class_id'],r)
rows=list(first.values());selected={r['clip_id'] for r in rows};rows.extend(r for r in ordered if r['clip_id'] not in selected);rows=rows[:2048]
class Reader(CalibrationDataReader):
 def __init__(self):self.rows=iter(rows)
 def get_next(self):
  r=next(self.rows,None);return None if r is None else from_features(np.load(data/r['features'],allow_pickle=False))
quantize_static(str(fp),str(target),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['Conv','MatMul','Gemm'])
q=ort.InferenceSession(str(target),o);cos=[]
for r in rows[:12]:
 feed=from_features(np.load(data/r['features'],allow_pickle=False));a=session.run(None,feed)[0];b=q.run(None,feed)[0];cos.append(float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
t=time.monotonic()
for _ in range(8):q.run(None,feed)
report=dict(model_bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),frames=32,input_profile='palm3d-context-v3 body/hand normalized XY + binary observed availability; ears/face omitted',training_embedding_cosine_mean=float(np.mean(cos)),training_embedding_cosine_min=float(min(cos)),batch1_cpu_seconds=(time.monotonic()-t)/8,calibration_train_ids=[r['clip_id'] for r in rows],test_read=False)
(root/'encoder-mp2-quantization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
