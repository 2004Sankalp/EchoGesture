# AI disclosure

OpenAI Codex assisted with interpreting supplied contest documents, researching primary model sources, implementing the app, geometry, training, export, validation, tests, documentation and presentation draft. The team must review these artifacts and supply its actual identity and demonstration.

The current 3D temporal models were trained/fine-tuned on the Mac CPU using public inferred-XYZ poses. Their run histories, initialization and selection evidence accompany the project. Training requires no cloud language model. The older OpenHands SL-GCN pretrained baseline remains separately credited and is not the current default classifier. OpenHands data metadata and OpenCV Zoo / Google MediaPipe pretrained landmark components are third-party work.

An additional ASLLVD training-only experiment uses exact normalized-gloss matches; it is included in the deployed model only if the final model card lists that checkpoint. No ASLLVD test clips are added to training. Published third-party accuracy numbers are not our results.

At runtime camera images stay local. No cloud LLM rewrites signs into sentences. The signer explicitly confirms captions. Confidence is an uncalibrated model score. This prototype has not been validated with Deaf users or certified as an accessibility tool.

The 70% top-1 target has not been reached. Cached public-pose scores, tiny real-video development diagnostics and desktop timing are reported separately. No official accuracy, phone latency, robustness, power or thermal pass is claimed without measurements. The presentation uses actual current results and intentionally leaves team/member placeholders for completion.


The 22 November raw-video identity audit supersedes the earlier pose-only cleanliness assumption. Final fresh ASL encoders and appearance heads exclude the newly identified overlaps and do not initialize from historical ASL checkpoints. The independent appearance encoder uses official Torchvision MobileNetV2 ImageNet QNNPACK INT8 weights; this upstream ImageNet training is not an ASL result. Clip fusion and any low-rank compression are measured approximations, not substitutes for accuracy evaluation. Human-triggered Android speech uses an installed offline English system voice; playback has not been phone-validated.


The 22 November transfer experiments use the independently pretrained publisher OpenASL pose-only encoder with fresh WLASL word heads. Earlier statements about entirely fresh ASL initialization apply to the from-scratch 3D candidates. AI assistance implemented the transfer wrapper, feature/statistics adapters, temporal adaptation experiments, validation alignment, and documentation. Publisher weights, licensing, source-ID screening and its limitations are disclosed in the third-party notices and development audit. No WLASL-pretrained checkpoint is used for the corrected-split candidates. Validation determines model/ensemble selection; no new final test predictions have been used for selection. Cached-pose experiments are distinct from raw-video and phone validation.
