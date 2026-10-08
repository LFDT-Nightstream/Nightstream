#!/usr/bin/env python3
"""Check the Lean-emitted artifact digests and the paths that reuse them.

Owns two fail-closed checks over the checked-out tree:
1. In each artifact directory below, `SHA256SUMS` lists every regular
   (not symbolically linked) `*.json` file exactly once, and every listed
   SHA-256 digest matches its file.
2. Outside `formal/nightstream-fprime/artifacts`, every tracked file that has
   the name or the bytes of an artifact there is a symbolic link to that
   artifact. A regular copy can drift, so it is rejected.

It does not regenerate an artifact or decide whether a digest change is
valid. After a regeneration, run it with `--write` and commit the changed
`SHA256SUMS` files with the artifacts.
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
SYMLINK_MODE = '120000'


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
    """Return the number of checked digests in `directory`."""
    label = directory.relative_to(ROOT)
    expected = read_manifest(directory / 'SHA256SUMS')
    present = set(regular_artifacts(directory))
    for name in sorted(present - expected.keys()):
        errors.append(f'{label}/{name}: not listed in SHA256SUMS')
    for name in sorted(expected.keys() - present):
        problem = 'is a symbolic link; list only regular files' if (directory / name).is_symlink() else 'is missing'
        errors.append(f'{label}/{name}: listed in SHA256SUMS but {problem}')
    checked = sorted(present & expected.keys())
    for name in checked:
        value = digest(directory / name)
        if value != expected[name]:
            errors.append(f'{label}/{name}: digest {value} does not match SHA256SUMS')
    return len(checked)


def tracked_entries():
    """Yield `(mode, blob, relative path)` for every tracked file."""
    output = subprocess.run(['git', 'ls-files', '-s', '-z'], cwd=ROOT, check=True,
                            capture_output=True).stdout.decode()
    for record in filter(None, output.split('\0')):
        metadata, relative = record.split('\t', 1)
        mode, blob, _ = metadata.split()
        yield mode, blob, relative


def check_reuses(errors):
    """Return the number of links that resolve to their Lean artifact."""
    entries = list(tracked_entries())
    lean_blobs = {blob: Path(relative).name for mode, blob, relative in entries
                  if mode != SYMLINK_MODE and relative.endswith('.json')
                  and (ROOT / relative).parent == LEAN_ARTIFACTS}
    lean_names = set(lean_blobs.values())
    links = 0
    for mode, blob, relative in entries:
        path = ROOT / relative
        if path.parent == LEAN_ARTIFACTS:
            continue
        name = path.name if path.name in lean_names else lean_blobs.get(blob)
        if name is None:
            continue
        target = LEAN_ARTIFACTS / name
        if mode != SYMLINK_MODE or path.resolve() != target.resolve():
            errors.append(f'{relative}: must be a symbolic link to {target.relative_to(ROOT)}')
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
    count = sum(check_directory(directory, errors) for directory in ARTIFACT_DIRECTORIES)
    links = check_reuses(errors)
    for error in errors:
        print(f'[artifacts] {error}', file=sys.stderr)
    if errors:
        return 1
    print(f'[artifacts] {count} digests match; {links} links resolve to their Lean artifacts')
    return 0


if __name__ == '__main__':
    sys.exit(main())
