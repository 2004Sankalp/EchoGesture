# EchoGesture

Local isolated ASL recognition with words the signer reviews and confirms.

**Research prototype with unmet evaluation targets.** The submitted INT8 ensemble measures **53.35% validation top-1** on 3,865 corrected-split public pose clips and **51.81% test top-1** on the unchanged 2,878 public test pose clips. It retains all **2,000 provisional WLASL outputs**. The **70% target is unmet**. These are inferred-pose diagnostics, not organiser video accuracy or phone accuracy. The larger research ensemble reached 62.72% validation top-1 and is **not the deployed model**.

College: **SRM Institute of Science and Technology**. Team name/member details are intentionally blank until the registered team fills them. See [team-leader handoff](START_HERE_TEAM_LEADER.txt) and the editable [nine-slide presentation](presentation/EchoGesture_Final.pptx).

## Run on a laptop

Use **Python 3.11**:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
python src/main.py /absolute/path/to/one-sign.mp4
python src/live.py
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\activate`. Internet is needed for dependency installation; inference uses the supplied local assets. No frames are uploaded or saved by the app. The desktop CPU landmark profile is different from Android Holistic and needs its own video evaluation.

Desktop controls: **R** starts/stops one sign; **1–5** confirms a candidate; **U** undoes; **C** clears; **Q** quits. This is isolated-gloss recognition, not continuous sentence translation or sign synthesis. Confidence is uncalibrated.

## Android and demonstration

[Install the ARM64 APK](releases/EchoGesture-debug.apk) on the realme GT 6T / Android 16. Allow camera, capture one isolated sign, review alternatives and confirm a word. The app includes caption HUD, Undo/Clear, typed reply and user-triggered speech using an installed offline English voice. Handset operation and voice playback have **not** been physically validated. The leader may test them if time permits; do not infer compatibility from build success.

[Watch the 2:15 software walkthrough](demo/EchoGesture-Walkthrough.mp4). It uses **public training-pose replay, actual local inference and scripted confirmations**. It is not live phone capture, an accuracy benchmark, or a user study. [Disclosure](demo/DEMO_DISCLOSURE.txt) and replay receipts are supplied. Prefer an additional real handset recording if available.

## Architecture and reproducibility

A 308-channel inferred-XYZ representation combines canonical palm shape, orientation/body context, wrist movement and explicit missingness. Two branches combine a CPU-trained 3D temporal model and an independently pretrained OpenASL graph encoder with a fresh word head. The graph encoder consumes XY and observed availability; actual inferred XYZ/finger statistics are separate classifier features. Canonical shape has numerical rigid-transform/positive-scale invariance; this does not establish full real-world viewpoint or occlusion robustness.

The corrected training cohort contains 13,771 WLASL clips plus 16,959 original ASL Citizen training clips. A raw-video SHA256 audit removed 17 further training and three validation clips beyond pose-hash screening. The test set remains intact. WLASL signer overlap and possible near-duplicates remain; Citizen screening has a different provenance scope. No signer-independent claim is made. No historical WLASL checkpoint or ASLLVD addition initializes the submitted models. Validation selects checkpoints, global weights and geometry views; the final test is not used for selection.

[Selected weights/conversion](training/README.md), [full reproduction](research/README.md), [source contract](src/README.md), [third-party notices](THIRD_PARTY_NOTICES.md) and [AI disclosure](AI_DISCLOSURE.md) are included. The exact calibrated transfer encoder is supplied for conversion, without duplicating its source language decoder. Original videos/pose archives are not redistributed.

## Evidence and limits

| Measure | Current evidence |
|---|---|
| Public-pose test top-1 | 51.81% on 2,878 clips; target70% unmet |
| Public-pose test top-5 | 81.79%; does not substitute for top-1 |
| Active Android inference assets | 19.276 MB, including Holistic, classifiers, labels and configuration |
| Active desktop inference assets | 13.185 MB; separate CPU landmark profile |
| Physical phone latency/power/thermal | Not measured |
| Paired video robustness | Not measured |
| Source checks | 42 passed; geometry parity recorded separately |

INT8-QDQ applies to classifier Conv/Gemm/MatMul where supported; other operators remain float. Android Holistic is FLOAT16. This is not a wholly integer pipeline. APK/native-runtime size is separate from model assets. The repository contains separate platform frontends and conversion-only artifacts; their combined distribution size is not one device's active model size.

The frozen [model card](models/model_card.json), [evidence PDF](docs/Current-Evidence.pdf), [test report](results/ensemble_test.json), [predictions](results/predictions.jsonl) and metric CSVs declare their measurement scopes. Historical experiments are explicitly separated under `results/history/`. No overall jury pass or winning outcome is claimed.

The leader must fill registered team details, publish these **extracted files** to a **public GitHub repository**, and have the primary member submit the form once. No repository or form has been published/submitted by the agent.
