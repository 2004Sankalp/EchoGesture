"""Compose the calibrated encoder unchanged with portable stats and INT8 head."""
import argparse,hashlib,json,copy
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType
from mp_preprocessor import MPGraphInputs
from portable_model import sequence_statistics,geometry_statistics
from head import GeometryPoseHead
from export_3d import FrozenCosine
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();torch.set_num_threads(1);base=Path('work/unisign-study');cache=base/'features-mp';data=Path('work/3d-data-citizen-v3');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
manifest=Path(a.manifest);manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
if state['manifest_sha256']!=manifest_sha:raise ValueError('Manifest mismatch')
encoder=base/'encoder-mp2.int8.onnx'
if hashlib.file_digest(encoder.open('rb'),'sha256').hexdigest()!=state['encoder_sha256']:raise ValueError('Encoder mismatch')
rows=json.loads(manifest.read_text());train=[r for r in rows if r['split']=='train'];ordered=sorted(train,key=lambda r:hashlib.sha256(r['clip_id'].encode()).hexdigest());first={}
for r in ordered:first.setdefault(r['class_id'],r)
selected=list(first.values());seen={r['clip_id'] for r in selected};selected.extend(r for r in ordered if r['clip_id'] not in seen);selected=selected[:2048]
# Original manifest records point to the original, immutable feature files.
x=np.load(data/selected[0]['features'],allow_pickle=False).astype(np.float32)[None]
pre=MPGraphInputs().eval();torch.onnx.export(pre,torch.from_numpy(x),str(out/'preprocess.onnx'),input_names=['features3d'],output_names=['body','left','right'],opset_version=17,dynamo=False,dynamic_axes={n:{0:'batch'} for n in ['features3d','body','left','right']})
class Statistics(torch.nn.Module):
 def forward(self,sequence,features):return torch.cat((sequence_statistics(sequence),geometry_statistics(features)),1)
torch.onnx.export(Statistics().eval(),(torch.randn(1,768,32),torch.from_numpy(x)),str(out/'statistics.onnx'),input_names=['sequence','features3d'],output_names=['head_features'],opset_version=17,dynamo=False,dynamic_axes={n:{0:'batch'} for n in ['sequence','features3d','head_features']})
head=GeometryPoseHead(3072,state['channels']);head.load_state_dict(state['model']);head.eval();head.classifier=FrozenCosine(head.classifier.weight);torch.onnx.export(head,torch.randn(1,4088),str(out/'head.fp32.onnx'),input_names=['head_features'],output_names=['logits'],opset_version=17,dynamo=False,dynamic_axes={'head_features':{0:'batch'},'logits':{0:'batch'}})
class Reader(CalibrationDataReader):
 def __init__(self):self.rows=iter(selected)
 def get_next(self):
  r=next(self.rows,None);return None if r is None else {'head_features':np.load(cache/(r['clip_id']+'.noface.npy'),allow_pickle=False).astype(np.float32)[None]}
quantize_static(str(out/'head.fp32.onnx'),str(out/'head.int8.onnx'),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['MatMul','Gemm'])
onnx.utils.extract_model(str(encoder),str(out/'encoder-pruned.int8.onnx'),['body','left','right'],['pose_features'])
models=[]
for prefix,path in [('prep_',out/'preprocess.onnx'),('enc_',out/'encoder-pruned.int8.onnx'),('stat_',out/'statistics.onnx'),('head_',out/'head.int8.onnx')]:models.append(onnx.compose.add_prefix(onnx.load(str(path)),prefix))
prep,enc,stats,h=models
links={prep.graph.input[0].name:'features3d',stats.graph.input[1].name:'features3d',stats.graph.input[0].name:enc.graph.output[0].name,h.graph.input[0].name:stats.graph.output[0].name,h.graph.output[0].name:'logits'}
for left,right in zip(prep.graph.output,enc.graph.input):links[right.name]=left.name
nodes=[];initializers=[];value_info=[]
for m in models:
 for node in m.graph.node:
  for i,v in enumerate(node.input):node.input[i]=links.get(v,v)
  for i,v in enumerate(node.output):node.output[i]=links.get(v,v)
 nodes.extend(m.graph.node);initializers.extend(m.graph.initializer)
graph=onnx.helper.make_graph(nodes,'EchoGesture-OpenASL-3D-portable',[onnx.helper.make_tensor_value_info('features3d',onnx.TensorProto.FLOAT,['batch',308,48])],[onnx.helper.make_tensor_value_info('logits',onnx.TensorProto.FLOAT,['batch',2000])],initializers)
model=onnx.helper.make_model(graph,opset_imports=[onnx.helper.make_opsetid('',17)],producer_name='EchoGesture');model.ir_version=min(m.ir_version for m in models);target=out/'recognizer3d.int8.onnx';onnx.checker.check_model(model);onnx.save(model,str(target));o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;session=ort.InferenceSession(str(target),o,providers=['CPUExecutionProvider']);actual=session.run(None,{'features3d':x})[0]
# Verify composition against the separately executed components.
feed={z.name:y.numpy() for z,y in zip(ort.InferenceSession(str(encoder),o).get_inputs(),pre(torch.from_numpy(x)))};sequence=ort.InferenceSession(str(encoder),o).run(None,feed)[0]
with torch.inference_mode():pooled=Statistics()(torch.from_numpy(sequence),torch.from_numpy(x)).numpy()
expected=ort.InferenceSession(str(out/'head.int8.onnx'),o).run(None,{'head_features':pooled})[0];np.testing.assert_allclose(actual,expected,rtol=.001,atol=.001)
meta=dict(feature_version='palm3d-context-v3',frames=48,input_features=308,classes=2000,architecture='independent-OpenASL-graph+true3Dstats',model_bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),dataset_manifest_sha256=manifest_sha,encoder_sha256=state['encoder_sha256'],calibration_clip_ids=[r['clip_id'] for r in selected],quantization='Original INT8 QDQ graph encoder preserved; INT8 QDQ classifier; preprocessing, normalization, geometry and statistics float',composition_max_abs_error=float(abs(actual-expected).max()),selection='Validation checkpoint only; merged INT8 validation still required; no phone or latency pass')
(out/'conversion3d.json').write_text(json.dumps(meta,indent=2));(out/'labels.json').write_text(json.dumps(state['labels'],indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='calibration_clip_ids'}),flush=True)
