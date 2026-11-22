# Third-party notices

## OpenHands

AI4Bharat/OpenHands, commit `b9ccf9eaf2a71301fc4601a64a6af06fdef04c9b`, Apache License 2.0. Full licence retained in `src/vendor/LICENSE-OpenHands.txt`. The two vendored modules originate from the graph encoder implementation. `decoupled_gcn.py` was modified to accept a plain dictionary configuration. Runtime export optimisation is implemented separately in `src/convert.py`.

Source: https://github.com/AI4Bharat/OpenHands

Weights: https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/wlasl_slgcn.zip

The optional legacy baseline uses those pretrained research weights; the current default uses separately CPU-trained EchoGesture temporal models. EchoGesture does not claim authorship of the third-party baseline. See the upstream release and WLASL data terms when redistributing derived research assets publicly. Repository source licensing alone does not replace dataset usage conditions.

## WLASL

Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison, Li et al. https://dxli94.github.io/WLASL/

Public pose archive linked by OpenHands: https://zenodo.org/records/6674324/files/WLASL.zip

The dataset's own usage terms apply. No raw dataset video or pose archive is included in this repository. The label list and calibration ID manifest are derived metadata. Published benchmark results are not represented as EchoGesture results.

## MediaPipe

Google MediaPipe code and Holistic landmark model, used under their upstream terms. https://github.com/google-ai-edge/mediapipe

Bundled Android task source: https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

## ONNX Runtime and platform dependencies

Microsoft ONNX Runtime, MIT. https://github.com/microsoft/onnxruntime

AndroidX CameraX and Activity, Apache-2.0. https://developer.android.com/jetpack/androidx

Python dependencies retain their respective upstream licences. Do not remove attribution from the final public repository.

## OpenCV Zoo CPU landmark profile

OpenCV Zoo commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`, Apache-2.0. Four converted MediaPipe block-quantized landmark model assets and credited wrappers are used by the desktop extractor. Licenses are retained under `src/vendor/opencv_zoo/`; exact upstream URLs and model hashes are in `models/opencv/provenance.json`. Wrappers were corrected to convert XYXY boxes to OpenCV NMS XYWH. https://github.com/opencv/opencv_zoo

## Additional ASLLVD training poses

ASLLVD is the American Sign Language Lexicon Video Dataset (Boston University ASLLRP). Original dataset: https://www.bu.edu/asllrp/av/dai-asllvd.html . OpenHands public derived pose archive: https://zenodo.org/records/6674324/files/ASLLVD-Skeleton.zip . The Zenodo record advertises CC-BY-4.0 for the pose dataset; original underlying materials retain their own terms. No original video or pose archive is redistributed here. Only original ASLLVD training split examples with exact normalized label matches were added; original ASLLVD test was unused. See `results/additional_training_data.json`.


## Raw-video identity audit and final training sources

The 22 November final training protocol omits ASLLVD additions; the earlier ASLLVD description applies to historical experiments only. Final fresh ASL encoders use the corrected WLASL training split and original ASL Citizen training clips with exact label matches. No dataset archive or original video is redistributed. ASL Citizen sources: https://github.com/microsoft/ASL-citizen-code and https://huggingface.co/datasets/SorensenAI/asl-citizen-poses . Original dataset terms remain applicable.

Uni-Sign provides WLASL RTMPose WholeBody XY and word-video archives used for identity auditing and the independent appearance experiment. Source: https://github.com/ZechengLi19/Uni-Sign ; data/model hosting: https://huggingface.co/ZechengLi19/Uni-Sign . The hosting repository identifies CC-BY-NC-4.0. The separate transfer candidates use the publisher’s OpenASL pose-only checkpoint, not its WLASL checkpoint. Its word-classification heads are initialized fresh on the corrected WLASL split. The from-scratch 3D candidates do not use this checkpoint. No underlying video/pose archive is redistributed.

## Torchvision appearance encoder

Torchvision 0.23.0, BSD-3-Clause software license retained in licenses/Torchvision-BSD-3-Clause.txt. The appearance experiment uses the official MobileNetV2 IMAGENET1K_QNNPACK_V1 ImageNet-pretrained quantization-aware weights, with the ImageNet classifier removed. Source, recipe and weights: https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.quantization.mobilenet_v2.html . Own ASL classifier heads are initialized fresh and trained on the corrected split. Torchvision software licensing does not replace underlying dataset/weight terms.


## Independent OpenASL transfer experiment

Uni-Sign / CoSign OpenASL pose-only checkpoint: https://huggingface.co/ZechengLi19/Uni-Sign/blob/main/openasl_pose_only_slt.pth . Hosting terms identify CC-BY-NC-4.0; preserve attribution and noncommercial restrictions if a derived model is included. Original checkpoint SHA256: `f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798`. Our inference implementation reconstructs the architecture from tensor names and published graph-convolution equations; it does not vendor the publisher Python modules or instantiate its language decoder.

OpenASL original dataset: https://github.com/chevalierNoir/OpenASL . Its original dataset terms remain applicable, including the stated CC-BY-NC-ND-4.0 terms. Source-ID screening found no overlaps between 2,042 OpenASL training source video IDs and parsed WLASL held-out YouTube URLs. This does not rule out mirrors or near-duplicates; non-YouTube and unparsed URLs were not covered. Upstream pretraining provenance remains based on publisher records. See `results/development/openasl_pretraining_audit.json`. The graph encoder uses XY and confidence or binary observation flags; true inferred XYZ and 3D finger kinematics remain separate inputs to the word head.
