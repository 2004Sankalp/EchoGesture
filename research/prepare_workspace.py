"""Set the documented relative workspace paths from frozen metadata."""
from pathlib import Path
import gzip,shutil
root=Path.cwd()
if not (root/'src/train_3d.py').exists():raise SystemExit('Run from the EchoGesture repository root')
for p in ['work/appearance','work/3d-data-v3','work/3d-data-citizen-v3']:(root/p).mkdir(parents=True,exist_ok=True)
raw=gzip.decompress((root/'resources/protocol/wlasl_manifest_video_clean.json.gz').read_bytes())
(root/'work/manifest-wlasl-video-clean.json').write_bytes(raw);(root/'work/3d-data-v3/manifest.json').write_bytes(raw)
shutil.copyfile(root/'models/labels.json',root/'work/3d-data-v3/labels.json')
# Selected transfer conversion uses the exact calibrated encoder supplied here.
transfer=root/'work/unisign-study';transfer.mkdir(parents=True,exist_ok=True)
for name in ['encoder-mp2.int8.onnx','encoder-mp2-quantization.json']:
    shutil.copyfile(root/'training/conversion'/name,transfer/name)
# Appearance is an optional historical trial, not part of selected inference.
if (root/'models/appearance/encoder.int8.onnx').exists():
    shutil.copyfile(root/'models/appearance/encoder.int8.onnx',root/'work/appearance/encoder.qat_views.onnx')
    shutil.copyfile(root/'models/appearance/encoder.json',root/'work/appearance/encoder-views.json')
