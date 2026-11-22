"""Read complete local ZIP entries, including a partial download, for calibration.

Only upstream train clips enter calibration. Test clips stay separate. A partial
archive is a convenience sample and cannot establish full benchmark accuracy.
Only plain NumPy pose structures are accepted from the historical pickle files.
"""
import argparse
import io
import json
import pickle
from pathlib import Path
import struct
import zlib
import numpy as np
from echogesture.features import prepare


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported type {module}.{name}')
        return allowed[module, name]


def entries(path):
    with Path(path).open('rb') as stream:
        while True:
            header = stream.read(30)
            if len(header) < 30 or header[:4] != b'PK\x03\x04':
                return
            _, _, flags, method, _, _, crc, compressed_size, _, name_size, extra_size = struct.unpack('<4s5H3I2H', header)
            name = stream.read(name_size).decode('utf-8')
            stream.read(extra_size)
            if flags & 8:
                raise ValueError('ZIP data descriptors unsupported; use a standard complete archive reader')
            compressed = stream.read(compressed_size)
            if len(compressed) != compressed_size:
                return
            raw = zlib.decompress(compressed, -15) if method == 8 else compressed
            if zlib.crc32(raw) & 0xffffffff != crc:
                raise ValueError('ZIP checksum mismatch')
            yield name, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    for folder in ('train', 'test'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.metadata).read_text())
    splits = {i['video_id']: (i['split'], entry['gloss'].upper()) for entry in data for i in entry['instances']}
    records = []
    for name, raw in entries(args.archive):
        clip_id = Path(name).stem
        if not name.endswith('.pkl') or clip_id not in splits:
            continue
        split, label = splits[clip_id]
        if split not in ('train', 'test'):
            continue
        points = ArrayUnpickler(io.BytesIO(raw)).load()['keypoints']
        prepared = prepare(points)
        np.save(out / split / (clip_id + '.npy'), prepared)
        records.append(dict(clip_id=clip_id, split=split, label=label, frames=int(prepared.shape[2])))
    (out / 'manifest.json').write_text(json.dumps(records, indent=2))
    print({split: sum(r['split'] == split for r in records) for split in ('train', 'test')})


if __name__ == '__main__':
    main()
