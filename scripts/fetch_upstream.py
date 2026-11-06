"""Download pinned reference repositories into ignored external/ directories."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / 'configs/upstream_sources.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('methods', nargs='+', choices=sorted(sources))
    args = parser.parse_args()
    (ROOT / 'external').mkdir(exist_ok=True)
    for name in args.methods:
        source = sources[name]
        target = ROOT / 'external' / source['directory']
        if target.exists():
            actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != source['commit']:
                raise RuntimeError(f'{target} exists at another commit; leaving it unchanged')
            print(f'Already present: {name} at {actual}')
            continue
        subprocess.run(['git', 'clone', source['url'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', source['commit']], check=True)
        print(f'Ready: {target}')


if __name__ == '__main__':
    main()
