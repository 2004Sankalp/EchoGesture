# Model selection notes

The current task keeps the declared 2,000-class vocabulary until an organiser vocabulary is confirmed. A smaller vocabulary result cannot establish a pass for this task.

## Checked candidates

- **OpenHands WLASL SL-GCN:** available pretrained weights and Apache-2.0 source. This is the attributed reference backend, not an original EchoGesture recognition architecture. Its checkpoint configuration uses the WLASL test split for validation. Together with raw-pose duplicate leakage in the published splits, this means its public test diagnostics are not a clean independent estimate. The new 3D training run starts from scratch and removes cross-split byte-identical poses.
- **PoET:** the official repository reports 82.2% per-class top-1 for WLASL100, 74.8% for WLASL300 and 46.1% for WLASL2000. These are different vocabulary sizes and a published per-class metric, not EchoGesture scores. The cloned source did not contain a licence file when checked. It is not vendored or used as the submitted model. Source: https://gitlab.lisn.upsaclay.fr/mtals/publications/poet
- **Google ASL Signs competition model:** the public model card describes a pretrained 250-class model. Its label file shares 201 exact uppercase labels with WLASL2000; that is insufficient coverage for the current 2,000-class task. It also expects face landmarks missing from the public 75-joint pose archive. It is not silently substituted for the full-vocabulary model. Model card: https://huggingface.co/sign/kaggle-asl-signs-1st-place . Original author source: https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
- **SignBart:** official code describes an efficient XY encoder/decoder and pretrained weights for multiple WLASL subsets. Its XY representation does not by itself meet the required 3D hand-feature requirement. It is not integrated. Source: https://github.com/TinhNguyen2312/SignBart
- **PhonSSM:** its README claims 72.08% for WLASL2000. Source inspection at commit `5169623e57939d08f71eb08aad94f448c89c737d` found that `training/benchmark_external.py` uses stratified random splits of preprocessed arrays, using the official split ratios rather than official clip membership. Its recorded test count is 8,634, versus 2,878 original WLASL2000 test clips. This is not a directly comparable official-split result and is not treated as a verified route to 70%. No PhonSSM score or weights are used as EchoGesture evidence. Source: https://github.com/anonymoussubmitter-167/phon-ssm

## EchoGesture 3D candidate

Palm-centred 3D hand shape, explicit hand validity, separate orientation/body context and wrist velocity feed a small temporal/channel MLP mixer. The canonical hand-shape calculation has numerical invariance tests. Body and hand depth reference frames remain separate. Inputs are inferred MediaPipe coordinates, not measured camera depth. Mathematical canonical-shape invariance does not prove robustness to real occlusion, camera angle or lighting.

Training uses the complete verified public pose archive, retains all 2,000 output classes, and selects checkpoints only on validation. The published test set stays intact. Signers overlap between the published splits, so no signer-independent result is claimed. Exact duplicate detection uses byte-level SHA-256 of decompressed pose records; it does not identify every near-duplicate or related source recording.

Model size, conversion, full-vocabulary accuracy and feature corruption diagnostics must all be measured after training. No published score is copied into EchoGesture results.


## Raw-video identity audit, 22 November

The newer Uni-Sign WLASL word-video archive exposes 20 additional byte-identical cross-split pairs (SHA256 verified). Pose-pickle hashes missed these because the extracted pose records differ. Exclude 17 training IDs and three validation IDs overlapping test, preserving the full 2,878-test-clip set and 2,000 output classes. Final fresh training uses 13,771 WLASL training clips plus 16,959 original ASL Citizen training clips; 3,865 validation clips. ASLLVD extras are omitted because equivalent raw-video identity verification is unavailable. Historical phase-1/phase-2 ASL checkpoints, teachers and scores are development artifacts; they are not clean final initialization or benchmark passes.

The appearance encoder is independently pretrained Torchvision MobileNetV2 QNNPACK INT8 on ImageNet. Its own classifiers start fresh on the corrected WLASL split. Global and spatial hand/body descriptors share one encoder. Crops use local landmarks, and missing hands are masked. The encoder is 2,419,825 bytes before exposing its feature-map output. RGB accuracy, combined model bytes, conversion and runtime still require measurement. No phone, power, robustness or thermal pass is claimed.

The raw-video identity audit applies to WLASL. ASL Citizen additions use the publisher’s original training membership and pose/feature identity screening against held-out WLASL; original ASL Citizen videos were not downloaded. This is not a claim of signer-independent evaluation or exhaustive near-duplicate detection.

Android source now includes explicit user-triggered speech for confirmed captions, selecting an installed English voice that does not require a network connection and does not advertise missing data. The Java/manifest changes build successfully. Audio playback and voice availability remain untested on a phone.


## Independent OpenASL transfer and corrected-split deployment

A publisher OpenASL pose-only encoder is being evaluated with newly initialized 2,000-word heads. No WLASL-pretrained checkpoint initializes these corrected-split candidates. Its graph channels are XY and confidence/availability; inferred XYZ and finger kinematics are separate classifier features. Source-ID screening covers parsed YouTube URLs only and cannot exclude every mirror or near-duplicate. Publisher pretraining provenance and restrictions are retained in the notices.

The corrected-split anatomy snapshot at epoch 53 measures 45.7697% float validation top-1 and 45.7439% INT8 ONNX validation top-1 over 3,865 cached public pose clips. Its recognizer is 1,258,519 bytes; this alone is not the complete inference asset budget. See `results/development/audited_anatomy_int8_validation.json`. Larger exploratory transfer ensembles are recorded separately and are not deployment, final-test, or live-video scores. No 70% pass has been demonstrated.
