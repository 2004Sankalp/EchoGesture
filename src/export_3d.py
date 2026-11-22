"""Export a validation-selected trained model; calibrate on train only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from echogesture.temporal3d import Temporal3D


class FrozenCosine(torch.nn.Module):
    """Fold fixed prototype normalisation before quantizing the classifier."""
    def __init__(self,weight):
        super().__init__()
        self.register_buffer('weight',torch.nn.functional.normalize(weight.detach(),dim=-1))

    def forward(self,x):
        return 20 * torch.nn.functional.linear(torch.nn.functional.normalize(x,dim=-1),self.weight)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--calibration-clips',type=int,default=2048)
    parser.add_argument('--manifest',help='Exact frozen training manifest; needed for audited snapshots')
    args=parser.parse_args();torch.set_num_threads(1);ort.disable_telemetry_events()
    root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    labels=json.loads((root/'labels.json').read_text())
    if labels!=state['labels']:raise ValueError('Vocabulary mismatch')
    torch.backends.mha.set_fastpath_enabled(False)
    model=Temporal3D(len(labels),state['channels'],state['frames'],state.get('input_features',state['model']['stem.weight'].shape[1]),state.get('classifier','linear'),state.get('architecture','mixer'));model.load_state_dict(state['model']);model.eval()
    trained_parameters=sum(p.numel() for p in model.parameters())
    manifest_bytes=(Path(args.manifest) if args.manifest else root/'manifest.json').read_bytes()
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    if state.get('dataset_manifest_sha256',manifest_hash)!=manifest_hash:raise ValueError('Conversion manifest differs from training snapshot')
    manifest=json.loads(manifest_bytes);train=[r for r in manifest if r['split']=='train']
    # Deterministic class-spread calibration sample, not handpicked by accuracy.
    train=sorted(train,key=lambda r:hashlib.sha256(r['clip_id'].encode()).hexdigest())
    first_per_class={}
    for row in train:first_per_class.setdefault(row['class_id'],row)
    selected=list(first_per_class.values())[:args.calibration_clips]
    selected_ids={row['clip_id'] for row in selected}
    selected.extend(row for row in train if row['clip_id'] not in selected_ids)
    train=selected[:args.calibration_clips]
    sample=np.load(root/train[0]['features'],allow_pickle=False).astype(np.float32)[None]
    with torch.inference_mode():expected=model(torch.from_numpy(sample)).numpy()
    if state.get('classifier')=='cosine':
        model.head[2]=FrozenCosine(model.head[2].weight)
        with torch.inference_mode():rewritten=model(torch.from_numpy(sample)).numpy()
        np.testing.assert_allclose(rewritten,expected,rtol=1e-5,atol=1e-5)
    fp=out/'recognizer3d.fp32.onnx';target=out/'recognizer3d.int8.onnx'
    torch.onnx.export(model,torch.from_numpy(sample),str(fp),input_names=['features3d'],output_names=['logits'],
        opset_version=17,dynamo=False,dynamic_axes={'features3d':{0:'batch'},'logits':{0:'batch'}})
    onnx.checker.check_model(str(fp))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(fp),options,providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None,{'features3d':sample})[0],expected,rtol=1e-4,atol=1e-4)
    class Reader(CalibrationDataReader):
        def __init__(self):self.rows=iter(train)
        def get_next(self):
            row=next(self.rows,None)
            return None if row is None else {'features3d':np.load(root/row['features'],allow_pickle=False).astype(np.float32)[None]}
    quantize_static(str(fp),str(target),Reader(),quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['Conv','Gemm','MatMul'])
    ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider']).run(None,{'features3d':sample})
    (out/'labels.json').write_text(json.dumps(labels,indent=2))
    meta=dict(feature_version=state['feature_version'],frames=state['frames'],classes=len(labels),channels=state['channels'],input_features=int(sample.shape[1]),classifier=state.get('classifier','linear'),architecture=state.get('architecture','mixer'),
        best_validation_top1=state['best'],best_epoch=state['best_epoch']+1,selection='validation only, test not consulted',
        calibration_clip_ids=[r['clip_id'] for r in train],model_bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),quantization='INT8-QDQ Conv/Gemm/MatMul; other operators float',
        parameters=trained_parameters,dataset_manifest_sha256=manifest_hash)
    (out/'conversion3d.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='calibration_clip_ids'},indent=2))


if __name__=='__main__':main()
