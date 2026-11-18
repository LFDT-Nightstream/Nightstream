#!/usr/bin/env python3
"""Check the Rocq sources of `Spec/Nebula/MemoryTrace.lean` against the SHA-256 pins of its
module docstring."""

import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / 'NightstreamFPrime/Spec/Nebula/MemoryTrace.lean'
ROCQ = ROOT.parent / 'wasm-rocq'
PIN = re.compile(r'^- `([A-Za-z0-9]+\.v)` SHA-256 `([0-9a-f]{64})`')


def main():
    pins = [PIN.match(line) for line in COPY.read_text().splitlines()]
    pins = [(match[1], match[2]) for match in pins if match]
    if {name for name, _ in pins} != {'MemoryTrace.v', 'R1CS.v'}:
        raise SystemExit(f'[rocq-copy] {COPY.name} must pin MemoryTrace.v and R1CS.v once each')
    failed = False
    for name, pinned in pins:
        actual = hashlib.sha256((ROCQ / name).read_bytes()).hexdigest()
        if actual != pinned:
            print(f'[rocq-copy] formal/wasm-rocq/{name} has SHA-256 {actual}, the copy pins {pinned}; '
                  'compare the copy with the source, then update the pin', file=sys.stderr)
            failed = True
    if failed:
        raise SystemExit(1)
    print('[rocq-copy] the Rocq sources of the memory-event copy are the pinned ones')


if __name__ == '__main__':
    main()
