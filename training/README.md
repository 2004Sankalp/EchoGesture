# Selected model reproduction

The supplied checkpoints contain selected weights, metadata and training history, without optimizer tensors. `anatomy.pt` is the from-scratch corrected-split 3D temporal model. `portable.pt` is the independently pretrained OpenASL transfer branch with a fresh 2,000-word classifier, trained on the corrected WLASL split plus original ASL Citizen training clips. No historical WLASL checkpoint or ASLLVD addition initializes these selected candidates.

Use Python 3.11 and `src/requirements-research.txt`. Follow `research/README.md` to reconstruct the frozen feature membership from the public archive and protocol metadata. The original publisher checkpoint is not redistributed because it exceeds GitHub's normal file limit; its URL, licensing and SHA256 are disclosed. For conversion-only reproduction, the exact calibrated graph encoder is supplied under `training/conversion/`; it is not an additional runtime model.

After reconstructing the frozen feature cache, copy the two `training/conversion/encoder-mp2*` files into `work/unisign-study/`, then run:

```sh
python research/pose_transfer/extract_mp.py
python research/pose_transfer/export_portable.py --checkpoint training/checkpoints/portable.pt --manifest work/unisign-study/features-mp/manifest.json --out work/export-portable
python src/export_3d.py --checkpoint training/checkpoints/anatomy.pt --data work/3d-data-citizen-v3 --manifest work/3d-data-citizen-v3/manifest-video-clean.json --output work/export-anatomy
```

The frozen public-pose validation determines global weights and geometry views in `models/ensemble/ensemble.json`. Do not choose a new configuration from the final test. Binary hashes can vary with exporter/runtime versions; the distributed runtime ONNX files and recorded hashes are authoritative. All calibration IDs are training-only. Reproducing feature extraction and training takes substantial CPU time; it is not required to run the submitted model.
