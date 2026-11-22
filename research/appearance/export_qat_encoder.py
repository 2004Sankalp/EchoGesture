"""Official Torchvision quantization-aware pretrained INT8 appearance encoder."""
import time,json,hashlib
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from torchvision.models.quantization import mobilenet_v2,MobileNet_V2_QuantizedWeights
root=Path('work/appearance');torch.set_num_threads(1);torch.hub.set_dir('work/torch-cache');torch.backends.quantized.engine='qnnpack'
m=mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,quantize=True).eval()
class Encoder(torch.nn.Module):
 def __init__(self):super().__init__();self.quant=m.quant;self.features=m.features;self.dequant=m.dequant
 def forward(self,x):return self.dequant(self.features(self.quant(x))).mean((2,3))
e=Encoder().eval();sample=torch.randn(1,3,160,160);target=root/'encoder.qat_int8.onnx'
with torch.inference_mode():expected=e(sample).numpy()
torch.onnx.export(e,sample,str(target),input_names=['images'],output_names=['embeddings'],opset_version=17,dynamo=False,dynamic_axes={'images':{0:'batch',2:'height',3:'width'},'embeddings':{0:'batch'}})
onnx.checker.check_model(str(target));o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;s=ort.InferenceSession(str(target),o,providers=['CPUExecutionProvider']);actual=s.run(None,{'images':sample.numpy()})[0];cos=float((actual*expected).sum()/(np.linalg.norm(actual)*np.linalg.norm(expected)+1e-9))
print(json.dumps({'onnx_torch_sample_cosine':cos,'max_absolute':float(np.max(np.abs(actual-expected)))}),flush=True)
if cos<.98:raise ValueError('Export differs from quantized PyTorch encoder')
bench=[]
for size in [96,160]:
 x=np.random.default_rng(7).normal(size=(8,3,size,size)).astype(np.float32);s.run(None,{'images':x});t=time.monotonic()
 for _ in range(5):s.run(None,{'images':x})
 bench.append(dict(size=size,batch=8,seconds=(time.monotonic()-t)/5))
report=dict(encoder='Torchvision0.23 MobileNetV2 official ImageNet1K_QNNPACK_V1; features only',features=1280,quantization='Official pretrained quantized Conv/linear/add; INT8 weights and activations, float preprocessing/output pooling',model_bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),onnx_torch_sample_cosine=cos,benchmark=bench,test_payloads_read=False,source='https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.quantization.mobilenet_v2.html')
(root/'encoder-qat.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

# Expose the pre-pooling map; the numerical encoder weights stay unchanged.
model=onnx.load(target);last=model.graph.node[-1]
assert last.op_type=="ReduceMean"
model.graph.output.append(onnx.helper.make_tensor_value_info(last.input[0],onnx.TensorProto.FLOAT,["batch",1280,"map_height","map_width"]))
views=root/"encoder.qat_views.onnx";onnx.checker.check_model(model);onnx.save(model,views)
report.update(sha256=hashlib.file_digest(views.open("rb"),"sha256").hexdigest(),model_bytes=views.stat().st_size,outputs="global embedding plus original dequantized feature map; no additional learned weights")
(root/"encoder-views.json").write_text(json.dumps(report,indent=2))
