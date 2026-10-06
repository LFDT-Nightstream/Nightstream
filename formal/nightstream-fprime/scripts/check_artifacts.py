#!/usr/bin/env python3
"""Check the Lean-emitted artifact digests and their Rust copies.

Owns two fail-closed checks over the checked-out tree:
1. In each artifact directory below, `SHA256SUMS` lists every regular
   (not symbolically linked) `*.json` file exactly once, and every listed
   SHA-256 digest matches its file.
2. Every tracked file outside `formal/nightstream-fprime/artifacts` that has
   the name of an artifact there is byte-identical to that artifact.

It does not regenerate an artifact or decide whether a digest change is
valid. Update `SHA256SUMS` in the same change that regenerates an artifact.
"""

import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                           cwd=Path(__file__).resolve().parent, check=True,
                           capture_output=True, text=True).stdout.strip())
LEAN_ARTIFACTS = ROOT / 'formal/nightstream-fprime/artifacts'
ARTIFACT_DIRECTORIES = [
    LEAN_ARTIFACTS,
    ROOT / 'crates/nightstream/artifacts',
    ROOT / 'crates/nightstream-fprime/artifacts',
]


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_manifest(path):
    entries = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        value, separator, name = line.partition('  ')
        if not separator or len(value) != 64 or '/' in name or name in entries:
            raise SystemExit(f'{path}:{number}: malformed or repeated entry')
        entries[name] = value
    return entries


def check_directory(directory, errors):
    """Return the checked digests of the regular JSON files in `directory`."""
    label = directory.relative_to(ROOT)
    expected = read_manifest(directory / 'SHA256SUMS')
    present = {path.name for path in directory.glob('*.json') if not path.is_symlink()}
    for name in sorted(present - expected.keys()):
        errors.append(f'{label}/{name}: not listed in SHA256SUMS')
    for name in sorted(expected.keys() - present):
        errors.append(f'{label}/{name}: listed in SHA256SUMS but missing')
    actual = {name: digest(directory / name) for name in sorted(present & expected.keys())}
    for name, value in actual.items():
        if value != expected[name]:
            errors.append(f'{label}/{name}: digest {value} does not match SHA256SUMS')
    return actual


def main():
    errors = []
    checked = [check_directory(directory, errors) for directory in ARTIFACT_DIRECTORIES]
    lean = checked[0]
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode().split('\0')
    copies = 0
    for relative in tracked:
        path = ROOT / relative
        if path.name not in lean or path.parent == LEAN_ARTIFACTS:
            continue
        copies += 1
        if digest(path) != lean[path.name]:
            errors.append(f'{relative}: differs from {LEAN_ARTIFACTS.relative_to(ROOT)}/{path.name}')

    for error in errors:
        print(f'[artifacts] {error}', file=sys.stderr)
    if errors:
        return 1
    count = sum(len(entries) for entries in checked)
    print(f'[artifacts] {count} digests match; {copies} copies are identical')
    return 0


if __name__ == '__main__':
    sys.exit(main())
