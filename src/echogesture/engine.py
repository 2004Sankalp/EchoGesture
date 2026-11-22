import json
import time
from pathlib import Path
import numpy as np
from .features import prepare, extract_video

ROOT = Path(__file__).resolve().parents[2]


class Recognizer:
    def __init__(self, model=None, vocabulary=None):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        model = Path(model or ROOT / 'models/recognizer.int8.onnx')
        if not model.is_file():
            raise FileNotFoundError(f'Missing trained model: {model}. See src/README.md')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        self.model_path = model
        self.labels = json.loads((ROOT / 'models/labels.json').read_text())
        if vocabulary:
            official = json.loads(Path(vocabulary).read_text())
            if not isinstance(official, list) or not official or len(official) != len(set(official)):
                raise ValueError('Vocabulary must be a nonempty unique JSON list')
            missing = set(official) - set(self.labels)
            if missing:
                raise ValueError(f'Untrained official vocabulary: {sorted(missing)}')
            self.allowed = np.array([self.labels.index(label) for label in official])
        else:
            self.allowed = np.arange(len(self.labels))
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, points):
        features = self.prepare(points)
        start = time.perf_counter_ns()
        logits = self.session.run(None, {self.input_name: features})[0][0]
        elapsed = (time.perf_counter_ns() - start) / 1e6
        # Keep probabilities over the full trained vocabulary. Restricted
        # vocabulary must not manufacture inflated confidence.
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        ranks = self.allowed[np.argsort(-probabilities[self.allowed], kind='stable')[:5]]
        return {'prediction': self.labels[int(ranks[0])],
                'top5': [self.labels[int(i)] for i in ranks],
                'confidence': float(probabilities[ranks[0]]), 'latency_ms': elapsed}

    @staticmethod
    def prepare(points):
        return prepare(points)

    def predict(self, path, clip_id=None):
        start = time.perf_counter_ns()
        points = extract_video(path)
        prediction = self.classify(points)
        prediction['clip_id'] = clip_id if clip_id is not None else Path(path).stem
        # Full clip processing, not disguised as single-frame latency.
        prediction['pipeline_ms'] = (time.perf_counter_ns() - start) / 1e6
        return prediction


class Recognizer3D(Recognizer):
    """Compatible trained 3D feature model, opt-in until performance reviewed."""
    def __init__(self, model=None, vocabulary=None):
        folder = ROOT / 'models/ensemble'
        info = json.loads((folder / 'anatomy.conversion.json').read_text())
        from .features3d import SUPPORTED_VERSIONS
        if info['feature_version'] not in SUPPORTED_VERSIONS:
            raise ValueError('3D model feature version mismatch')
        self.frames = info['frames']
        self.feature_version=info['feature_version']
        super().__init__(model or folder / 'anatomy.int8.onnx', vocabulary)
        if self.labels != json.loads((folder / 'labels.json').read_text()):
            raise ValueError('3D label ordering differs from baseline labels')

    def prepare(self, points):
        from .features3d import prepare3d
        return prepare3d(points, self.frames,self.feature_version)[None]


class EnsembleRecognizer(Recognizer):
    """Frozen, validation-selected ensemble over all 2,000 trained signs."""
    def __init__(self, vocabulary=None, directory=None):
        import onnxruntime as ort
        folder = Path(directory) if directory is not None else ROOT / 'models/ensemble'
        self.config = json.loads((folder/'ensemble.json').read_text())
        self.members = self.config['members']
        if not self.members or not np.isclose(sum(m['weight'] for m in self.members), 1):
            raise ValueError('Ensemble weights must sum to one')
        self.model_paths = [folder/m['model'] for m in self.members]
        super().__init__(self.model_paths[0], vocabulary)
        if self.labels != json.loads((folder/'labels.json').read_text()):
            raise ValueError('Ensemble vocabulary differs')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.sessions = [self.session] + [ort.InferenceSession(str(p), options, providers=['CPUExecutionProvider']) for p in self.model_paths[1:]]
        self.model_asset_bytes = sum(p.stat().st_size for p in self.model_paths) + sum(p.stat().st_size for p in (ROOT/'models/opencv').glob('*.onnx'))

    def classify(self, points):
        from .features3d import prepare3d
        from .geometry import stretch_vertical
        start = time.perf_counter_ns()
        cache, logits = {}, None
        for member, session in zip(self.members, self.sessions):
            key = (member['feature_version'], member['frames'])
            if key not in cache:
                cache[key] = prepare3d(points, key[1], key[0])
            x = cache[key]
            views = [x if factor == 1 else stretch_vertical(x, factor) for factor in member['scales']]
            average = sum(session.run(None, {'features3d': view[None]})[0][0] for view in views)/len(views)
            weighted = average*member['weight']
            logits = weighted if logits is None else logits+weighted
        elapsed = (time.perf_counter_ns()-start)/1e6
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        ranks = self.allowed[np.argsort(-probabilities[self.allowed], kind='stable')[:5]]
        return {'prediction': self.labels[int(ranks[0])], 'top5': [self.labels[int(i)] for i in ranks],
                'confidence': float(probabilities[ranks[0]]), 'latency_ms': elapsed}


def create_recognizer(backend='ensemble', vocabulary=None):
    if backend == 'baseline':
        return Recognizer(vocabulary=vocabulary)
    if backend == '3d':
        return Recognizer3D(vocabulary=vocabulary)
    if backend == 'ensemble':
        return EnsembleRecognizer(vocabulary=vocabulary)
    raise ValueError('Backend must be baseline, 3d or ensemble')
