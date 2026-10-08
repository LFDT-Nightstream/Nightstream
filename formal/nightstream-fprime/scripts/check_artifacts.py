#!/usr/bin/env python3
"""Check the Lean-emitted artifact digests and the paths that reuse them.

Owns two fail-closed checks over the checked-out tree:
1. In each artifact directory below, `SHA256SUMS` lists every regular
   (not symbolically linked) `*.json` file exactly once, and every listed
   SHA-256 digest matches its file.
2. Outside its own directory, every tracked file that has the name or the
   bytes of one of these artifacts is a symbolic link to that artifact. A
   regular copy can drift, so it is rejected. Bytes are compared with the
   checked-out content, so Git LFS artifacts are covered too.

It does not regenerate an artifact or decide whether a digest change is
valid. After a regeneration, run it with `--write` and commit the changed
`SHA256SUMS` files with the artifacts.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                           cwd=Path(__file__).resolve().parent, check=True,
                           capture_output=True, text=True).stdout.strip())
ARTIFACT_DIRECTORIES = [
    ROOT / 'formal/nightstream-fprime/artifacts',
    ROOT / 'crates/nightstream/artifacts',
    ROOT / 'crates/nightstream-fprime/artifacts',
]


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def regular_artifacts(directory):
    return sorted(path.name for path in directory.glob('*.json') if not path.is_symlink())


def write_manifest(directory):
    lines = [f'{digest(directory / name)}  {name}\n' for name in regular_artifacts(directory)]
    (directory / 'SHA256SUMS').write_text(''.join(lines))


def read_manifest(path):
    entries = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        value, separator, name = line.partition('  ')
        if not separator or len(value) != 64 or '/' in name or name in entries:
            raise SystemExit(f'{path}:{number}: malformed or repeated entry')
        entries[name] = value
    return entries


def check_directory(directory, errors):
    """Return the SHA-256 digest of each regular artifact in `directory`."""
    label = directory.relative_to(ROOT)
    expected = read_manifest(directory / 'SHA256SUMS')
    actual = {name: digest(directory / name) for name in regular_artifacts(directory)}
    for name in sorted(actual.keys() - expected.keys()):
        errors.append(f'{label}/{name}: not listed in SHA256SUMS')
    for name in sorted(expected.keys() - actual.keys()):
        problem = 'is a symbolic link; list only regular files' if (directory / name).is_symlink() else 'is missing'
        errors.append(f'{label}/{name}: listed in SHA256SUMS but {problem}')
    for name in sorted(actual.keys() & expected.keys()):
        if actual[name] != expected[name]:
            errors.append(f'{label}/{name}: digest {actual[name]} does not match SHA256SUMS')
    return {directory / name: value for name, value in actual.items()}


def tracked_paths():
    output = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True,
                            capture_output=True).stdout.decode()
    return [ROOT / relative for relative in output.split('\0') if relative]


def check_reuses(artifacts, errors):
    """Return the number of links that resolve to their artifact.

    A name or a content belongs to the first directory in
    `ARTIFACT_DIRECTORIES` that holds it, so a same-name file in a later
    artifact directory is a reuse too."""
    by_name, by_content = {}, {}
    for path, value in artifacts.items():
        by_name.setdefault(path.name, path)
        by_content.setdefault((path.stat().st_size, value), path)
    sizes = {size for size, _ in by_content}
    links = 0
    for path in tracked_paths():
        if not os.path.lexists(path):
            continue
        target = by_name.get(path.name)
        if target is None and not path.is_symlink() and path.stat().st_size in sizes:
            target = by_content.get((path.stat().st_size, digest(path)))
        if target is None or target == path:
            continue
        if not path.is_symlink() or path.resolve() != target.resolve():
            errors.append(f'{path.relative_to(ROOT)}: must be a symbolic link to {target.relative_to(ROOT)}')
        else:
            links += 1
    return links


def main():
    arguments = sys.argv[1:]
    if arguments not in ([], ['--write']):
        raise SystemExit('usage: check_artifacts.py [--write]')
    if arguments:
        for directory in ARTIFACT_DIRECTORIES:
            write_manifest(directory)

    errors = []
    artifacts = {}
    for directory in ARTIFACT_DIRECTORIES:
        artifacts.update(check_directory(directory, errors))
    links = check_reuses(artifacts, errors)
    for error in errors:
        print(f'[artifacts] {error}', file=sys.stderr)
    if errors:
        return 1
    print(f'[artifacts] {len(artifacts)} digests match; {links} links resolve to their artifacts')
    return 0


if __name__ == '__main__':
    sys.exit(main())
