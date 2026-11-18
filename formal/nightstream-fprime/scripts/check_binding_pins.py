#!/usr/bin/env python3
"""Check that the three pins of the WASM binding exports (`SPEC.md` section 7.6, WASM binding
export) name the same digests: `scripts/wasm-binding.sha256`, the Rocq pin
`formal/wasm-rocq/scripts/binding-exports.sha256` and the Rust test
`crates/neo-wasm/tests/wasm_binding_export.rs`."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / 'scripts/wasm-binding.sha256'
ROCQ = ROOT.parent / 'wasm-rocq/scripts/binding-exports.sha256'
RUST = ROOT.parents[1] / 'crates/neo-wasm/tests/wasm_binding_export.rs'
NAMES = {f'wasm-binding-{size}-v1.json' for size in ('micro', 'small', 'medium', 'large')}


def sums(path):
    pins = dict(reversed(line.split('  ')) for line in path.read_text().splitlines())
    if set(pins) != NAMES:
        raise SystemExit(f'[binding-pins] {path} must pin the four exports once each')
    return pins


def main():
    lean, rocq = sums(LEAN), sums(ROCQ)
    rust = dict(re.findall(r'"(wasm-binding-[a-z]+-v1\.json)",\s*"([0-9a-f]{64})"', RUST.read_text()))
    if lean != rocq or lean != rust:
        raise SystemExit('[binding-pins] the Lean, Rocq and Rust pins of the binding exports differ')
    print('[binding-pins] the Lean, Rocq and Rust pins of the binding exports agree')


if __name__ == '__main__':
    main()
