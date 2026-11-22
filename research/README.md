# Reproduce the selected protocol

Use Python 3.11 from the project root and install `src/requirements-research.txt`. Download `work/WLASL-full.zip` from https://zenodo.org/records/6674324/files/WLASL.zip ; publisher MD5 `c45b690a5340da71d9a75db19e0cc572`. Original videos/pose archives are not redistributed. Frozen metadata under `resources/protocol/` fixes exact clip membership, feature checksums, labels and order.

```sh
python research/prepare_workspace.py
python src/prepare_frozen_features.py --archive work/WLASL-full.zip --output work/3d-data-citizen-v3
python src/cache_training_tensors.py --data work/3d-data-citizen-v3 --manifest work/3d-data-citizen-v3/manifest-video-clean.json --output work/3d-data-citizen-v3/shared-video-clean
```

The preparation step downloads the exact matched original ASL Citizen training poses from source URLs in the frozen metadata. It verifies source and feature hashes. Two no-hand examples are excluded. It uses no Citizen validation/test examples. Final membership is 13,771 WLASL training clips plus 16,959 Citizen training clips, 3,865 validation clips and all 2,878 original test clips. The raw-video identity audit removes 17 further training and three validation records beyond pose-hash cleanup. Signer overlap and possible near-duplicates remain. Citizen screening is not an exhaustive raw-video audit.

Selected weights are under `training/checkpoints/`; exact conversion steps are in `training/README.md`. The workspace helper copies the supplied calibrated transfer encoder and its provenance into the expected conversion workspace. For retraining the 3D candidates use `research/train_video_clean.py`. It selects validation only and saves complete-epoch best/last checkpoints atomically; interrupting the parent signals its children. Windows users may run the printed individual training commands directly.

The portable transfer branch reuses the publisher OpenASL pose-only graph encoder, not a WLASL checkpoint. Source URL: https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/openasl_pose_only_slt.pth ; SHA256 `f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798`. The original large checkpoint is not bundled. It is required for fresh encoder conversion/training, but the exact calibrated encoder is already supplied for conversion-only reproduction. Licensing and provenance limitations are in THIRD_PARTY_NOTICES.md and the development audit.

```sh
python research/pose_transfer/extract_mp.py
python research/pose_transfer/train_mp_controller.py
```

The controller trains fresh WLASL-only word heads, then initializes full-cohort heads from their own corrected-split checkpoints. It never uses historical WLASL weights or test-based selection. Graph inputs are XY and availability, not depth. Inferred XYZ/finger statistics remain separate head features. `mp_preprocessor.py` matches the NumPy cache adapter, and the composed exporter checks component parity, encoder/manifest hashes and training-only calibration IDs.

Historical appearance, fusion and other helpers may be retained as source context, but they are not the selected runtime recipe. Appearance weights are not bundled; that optional historical experiment requires its own export. The larger How2Sign/OpenASL research blend uses cached RTMPose inputs and is not the deployed model. No published score is copied into submission evidence.

Numeric/exporter differences can affect byte hashes on another environment. Distributed ONNX files and their receipts are authoritative. Candidate selection uses validation; do not tune to the final test or narrow the provisional 2,000-word task to claim 70%. Hardware latency, power, thermal and paired-video robustness require separate real-device evidence.
