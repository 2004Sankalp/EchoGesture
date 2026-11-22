"""Reproducible CPU export and quantization of OpenHands WLASL SL-GCN.

Requires a trusted upstream checkpoint, metadata and its config. Calibration is
optional: without real calibration data, export weight-only quantization and
explicitly label it mixed precision. Never calibrate on synthetic poses and
represent the result as a verified full INT8 solution.
"""
import argparse
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
from vendor.decoupled_gcn import DecoupledGCN, DecoupledGCNUnit


EDGES = [[2,0],[1,0],[0,3],[0,4],[3,5],[4,6],[5,7],[6,17],
         [7,8],[7,9],[9,10],[7,11],[11,12],[7,13],[13,14],[7,15],[15,16],
         [17,18],[17,19],[19,20],[17,21],[21,22],[17,23],[23,24],[17,25],[25,26]]


class FC(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classifier = nn.Linear(256, classes)

    def forward(self, x):
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = DecoupledGCN(2, {'num_nodes': 27, 'inward_edges': EDGES})
        self.decoder = FC(classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model(checkpoint, classes):
    # Historical Lightning callback names occur as dictionary keys. Map them
    # to inert local types; do not import or execute pickled external classes.
    class CheckpointKey: pass
    class EarlyStoppingKey: pass
    safe = [(CheckpointKey, 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
            (EarlyStoppingKey, 'pytorch_lightning.callbacks.early_stopping.EarlyStopping')]
    with torch.serialization.safe_globals(safe):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = data.get('state_dict', data)
    state = {key.removeprefix('model.'): value for key, value in state.items()}
    model = Model(classes)
    model.load_state_dict(state, strict=True)
    return model.eval()


def optimized_gcn(self, x0):
    # Exact frozen-adjacency rewrite: Einsum -> Conv and broadcast MatMul.
    # No learned parameter changes, checked against upstream before export.
    x = torch.nn.functional.conv2d(x0, self.export_weight, self.linear_bias.flatten())
    x = self.bn0(x)
    n, _, t, v = x.shape
    x = x.reshape(n, self.num_subset, self.out_channels // self.groups, self.groups, t, v)
    x = torch.matmul(x, self.export_adjacency.unsqueeze(1)).sum(1)
    x = x.reshape(n, self.out_channels, t, v)
    return self.relu(self.bn(x) + self.down(x0))


def freeze_graph(model):
    for module in model.modules():
        if isinstance(module, DecoupledGCNUnit):
            with torch.no_grad():
                adj = module.decoupled_A.repeat(1, module.out_channels // module.groups, 1, 1)
                adj = torch.cat([module.norm(adj[k:k+1]) for k in range(3)], dim=0)
                module.register_buffer('export_adjacency', adj[:, :module.groups].contiguous())
                module.register_buffer('export_weight', module.linear_weight.T[:, :, None, None].contiguous())
            module.forward = types.MethodType(optimized_gcn, module)


def main():
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, QuantFormat, CalibrationDataReader
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metadata', required=True, help='Upstream asl2000.json, never reordered by frequency')
    parser.add_argument('--output', default='models')
    parser.add_argument('--calibration', help='Directory of real prepared [1,2,T,27] .npy training tensors')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.metadata).read_text())
    original = sorted(entry['gloss'] for entry in entries)
    labels = [label.upper() for label in original]
    if len(labels) != len(set(labels)):
        raise ValueError('Uppercasing causes ambiguous labels')
    model = load_model(args.checkpoint, len(labels))
    sample = torch.randn(1, 2, 32, 27)
    with torch.no_grad():
        expected = model(sample).numpy()
        freeze_graph(model)
        actual = model(sample).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    fp = out / 'recognizer.fp32.onnx'
    torch.onnx.export(model, sample, str(fp), input_names=['poses'], output_names=['logits'],
        dynamic_axes={'poses': {2: 'frames'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(fp))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(fp), options, providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None, {'poses': sample.numpy()})[0], expected, rtol=1e-3, atol=1e-3)
    target = out / 'recognizer.int8.onnx'
    if args.calibration:
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.files = iter(sorted(Path(args.calibration).glob('*.npy')))
            def get_next(self):
                file = next(self.files, None)
                return None if file is None else {'poses': np.load(file, allow_pickle=False).astype(np.float32)}
        if not list(Path(args.calibration).glob('*.npy')):
            raise ValueError('Real training-only calibration tensors required')
        quantize_static(str(fp), str(target), Reader(), quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=False, op_types_to_quantize=['Conv','MatMul','Gemm'])
        mode = 'static INT8 QDQ Conv/MatMul/Gemm; other operators may remain float'
    else:
        quantize_dynamic(str(fp), str(target), weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul','Gemm'], per_channel=False)
        mode = 'dynamic INT8 MatMul/Gemm only; Conv remains FP32; not full INT8'
    # Fail if the converted graph cannot run.
    quantized = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    result = quantized.run(None, {'poses': sample.numpy()})[0]
    if not np.isfinite(result).all():
        raise ValueError('Quantized graph produced nonfinite logits')
    (out / 'labels.json').write_text(json.dumps(labels, indent=2))
    import hashlib
    report = {'classes': len(labels), 'quantization': mode,
              'model_bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'upstream_rewrite_max_abs_error': float(np.max(np.abs(expected - actual))),
              'synthetic_smoke_test_only': True, 'accuracy_verified': False}
    (out / 'conversion.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
