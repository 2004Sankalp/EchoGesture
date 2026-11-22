"""Evaluator entry point: Python 3.11, fully local video -> isolated gloss."""
import argparse
import json
import os
from functools import lru_cache
from echogesture.engine import create_recognizer


@lru_cache(maxsize=1)
def recognizer():
    return create_recognizer(os.environ.get('ECHOGESTURE_BACKEND', 'ensemble'),
                             vocabulary=os.environ.get('ECHOGESTURE_VOCABULARY'))


def predict(video_path: str) -> dict:
    """Default ID is the filename stem. Batch evaluation preserves manifest IDs."""
    result = recognizer().predict(video_path)
    return {key: result[key] for key in ('clip_id', 'prediction', 'top5', 'confidence', 'latency_ms')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    print(json.dumps(predict(args.video), allow_nan=False))
