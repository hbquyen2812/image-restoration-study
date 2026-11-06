"""Download the exact 64-image diagnostic subset and verify recorded SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/Train400_subset64')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/train400_subset64_manifest.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    for row in manifest['files']:
        target = args.out / row['file']
        if target.exists():
            data = target.read_bytes()
        else:
            request = urllib.request.Request(row['source'], headers={'User-Agent': 'image-restoration-study'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {target}; existing files are not overwritten')
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
        print(f'Verified {target.name}')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
