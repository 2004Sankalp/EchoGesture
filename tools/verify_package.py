"""Verify the frozen handoff files; no network or model execution required."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text());errors=[]
for name,entry in manifest.items():
 p=root/name
 if not p.is_file():errors.append('Missing '+name);continue
 if p.stat().st_size!=entry['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=entry['sha256']:errors.append('Changed '+name)
if errors:raise SystemExit('\n'.join(errors))
print(f'PASS: {len(manifest)} packaged files match frozen SHA256 hashes')
