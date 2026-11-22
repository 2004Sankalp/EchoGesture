# Track 2 requirements and evidence audit

Status: development prototype. No overall PASS or jury score is claimed. The 22 November raw-video audit supersedes earlier pose-hash-only accuracy claims; historical prototype figures below are not clean final benchmark passes. Corrected-split candidate evidence is under `results/development/`. Phone validation has not been performed. The team leader may test the handset after handoff if time permits, per the user’s latest instruction.

## Source precedence

1. User-supplied updated email and the current final form: submission 22 November 2026, 11:59 pm. The form does not state a timezone. Use the organiser's timezone and submit early; the working assumption here is India time.
2. Updated `XR_Hackathon_2026_Consolidated_Slides_V3-2.pptx`: Track 2 slides 8–11 and global submission slides 23–24.
3. `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf`: pages 9–12 for Track 2, including the exact source/results layout and Python 3.11. Its earlier general dates are superseded by the updated email/deck/form.
4. `Submission_Final_forGithub.pptx`: nine-slide structure for final presentation.
5. User screenshot: form checklist and primary-member-only submission. Live form checked read-only on 20 November 2026.

The attached `Student package_track04.zip` belongs to Track 4. Its action classes, manifest and JSON examples do not define Track 2 requirements. Document instructions are treated as contest specifications, not as authority to send emails, publish data, or submit a form.

Both shared final-submission short links resolve to the same form:

https://docs.google.com/forms/d/e/1FAIpQLSdko_96SLQAIoSnqRM6887NJQBHUfG0ay1KK1ORTdjXYkn4Bg/viewform

The live college option is `SRM Institute of Science and Technology`. The theme option is `EchoGesture: Sign Language Prediction on Edge Device`. No form has been filled or submitted. Team name is deliberately unset; registration and team identity remain unverified. The team naming pattern is `CollegeName_TeamName`, max four members from one college, one theme and one submission per team.

Updated milestones from the supplied email: top 12 announcement 29 November, final demo 9 December, results 12 December. Stale top-15, 18-November, 2-December and five-minute-demo wording remains in some slides and does not override the updated guidance. Use a 2–3 minute submission demo to satisfy the stricter Track 2 and global maximum-three-minute statements.

## Current jury evidence

Model quantization20%: mixed INT8 classifier graphs and exact conversions supplied; Holistic FLOAT16. Edge latency20%: physical phone not measured. Top1 accuracy35%: see the final model card;70% unmet. Robustness/HUD25%: HUD/review/Undo/Clear/typed reply/offline Speak implemented and compiled, but paired robustness/user validation not measured. No jury award or overall pass is claimed.

## Numeric targets

| Metric | Document target | Conservative acceptance rule | Evidence required |
|---|---|---|---|
| Top-1 accuracy | >=70% | >=70% on legitimate held-out evaluation with the full fixed vocabulary | Exact-ID predictions and labels, declared split, leakage audit |
| p95 latency | <=50 ms in table; <50 ms in jury text | <50 ms | Target-device timings, runtime, warmup policy, input length, stage scope and full pipeline separately |
| Model size | <=25 MB in table; <=20 MB / <20 MB elsewhere | <20,000,000 bytes | Artifact byte count, SHA-256, quantization inspection; disclose all feature models too |
| Robustness loss | <=5 percentage points | <=5 pt | Paired clean/perturbed videos with predefined low-light, occlusion, angle and motion conditions |
| Average power | <=500 mW | <=500 mW with stated methodology | Actual power measurement, idle baseline and instrument method; battery-current snapshots alone are insufficient |
| Thermal drift over 10 min | <=20% | <=20%, with a predefined measurement method | At least 10 min target-device run, initial/final timing windows and thermal logs |

Current active Android assets: 19.276143MB. Current active desktop assets: 13.185199MB. Exact hashes and all scopes are in `models/model_card.json`. Separate platform frontends and conversion artifacts are not loaded together. APK size is separate.

## Functional requirements

| Requirement | Implementation/evidence | Gap |
|---|---|---|
| Exocentric live smartphone camera | Android rear-camera source and local recognition workflow | Actual handset camera, orientation and compatibility test pending |
| Local processing, zero cloud streaming | Android manifest removes INTERNET; local model assets; no video-writing code or analytics in app code | Verify merged APK permissions and run in airplane mode on phone; dependency-level network trace not yet collected |
| Pose-invariant 3D hand features | Trained 308-feature palm-centered 3D pipeline; numerical rigid-transform invariance tests and Java/Python parity | Full pipeline retains orientation; real viewpoint/occlusion robustness not verified |
| Text or audio for two-way conversation | Confirmed captions, typed replies, installed offline English voice selected for user-triggered speech; build verified only | No validated continuous sentence translation; no sign synthesis |
| Python `predict(video_path)` | `src/main.py` returns required keys, uppercase word and top5 | Official vocabulary and clip-ID convention not received |
| Fixed vocabulary | Optional official vocabulary validator rejects untrained labels | Current 2,000 WLASL labels are provisional |
| Every clip exactly once | Strict batch validation | Official manifest and test clips unavailable |
| Reproducible source under `src/` | Python 3.11 pins, conversion and Android source | Target-device build/runtime validation pending |

## Required final submission artifacts

- Public GitHub repository with source, resources, documentation, presentation, demo and package. Not yet published.
- `team_name.zip` with `src/requirements.txt`, `src/main.py`, `src/README.md`, and `results/results.csv`.
- Both `results/predictions.jsonl` (normative per-clip lines) and `results/predictions.json` (layout/example requirement). They must contain the same actual predictions. Neither is fabricated here.
- Quantized model and conversion script. Included, with limitations documented.
- Actual demo video 2–3 minutes, max three minutes globally, or deployed app URL where accepted. Not yet recorded.
- Presentation named `CollegeName_TeamName`, following the supplied template. Not yet finalised because performance and team details remain incomplete.
- AI disclosure. Included.
- APK/SDK if supplied. Only claim a built APK when its build has succeeded; actual phone operation remains a separate check.

## Remaining submission work

1. Team fills its actual name/member details and verifies registration. All organiser context provided by the user has been used; no Track 2 vocabulary/test files were supplied. WLASL2000 remains an explicit provisional assumption.
2. The 70% top-1 target is unmet. Do not narrow the vocabulary, relabel test samples or use top-5 as top-1.
3. If time permits before submission, the leader may validate the selected APK on the realme. Actual handset compatibility, video accuracy, combined compute, paired robustness, measured app power and ten-minute thermal evidence remain pending. Android testing was deferred at the user's request.
4. Record the real 2–3 minute demonstration and replace presentation team/member placeholders. The supplied demo script is a recording guide, not a fabricated demonstration.
5. Publish the reviewed source/resources/APK/presentation/demo to a public GitHub repository; primary member submits the final form once. The form and repository have not been published/submitted by this agent.
