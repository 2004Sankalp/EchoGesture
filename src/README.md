# Evaluator and reproduction

Python 3.11. From the repository root, install `src/requirements.txt` and run `python src/main.py /path/to/one-sign.mp4`. `src/main.py:predict(video_path: str) -> dict` returns `clip_id`, uppercase `prediction`, `top5`, `confidence` and `latency_ms`. The default ID is the filename stem. For an organiser manifest, use the batch adapter to preserve exact IDs.

Each JSONL input row uses `clip_id`, `video_path` and optional `label`; relative paths resolve beside the manifest. This adapter is supplied by us, not asserted to be an organiser schema. Run:

```sh
python src/evaluate.py --manifest data/manifest.jsonl --vocabulary models/labels.json --device "ACTUAL DEVICE / OS" --output results/your_video_run
```

The default retains all2,000 provisional labels. Narrowing vocabulary changes the task and must not manufacture a70% pass. Outputs include predictions.jsonl, predictions.json, results.csv and separate pipeline timings. Classifier latency includes 3D preparation, fixed views and ensemble; video decoding and landmark extraction are separately timed. Public precomputed-pose diagnostics exclude extraction and are named accordingly.

The default ensemble and optional `3d` backend use supplied assets. The historical `baseline` backend is source-only and its old weights are not part of the final runtime package. For selected training/checkpoints and conversion use `training/README.md`; for full preparation use `research/README.md`. The old ASLLVD/legacy experiment helpers are retained for source context and are not the selected recipe.

Run `python -m pytest tests -q`. 42 checks passed in the handoff tree. The Java/Python geometry parity receipt is a desktop JVM check, not a physical-phone benchmark. Camera preview needs OS permission and a display. Desktop uses credited OpenCV CPU landmarks; Android uses Holistic Tasks. Both are distinct from the public archive profile.
