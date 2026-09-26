#!/usr/bin/env bash
# Compare Rust primitives with the committed Lean results; no Lean execution.
set -euo pipefail
parity_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$parity_root"
parity_dir="$(mktemp -d "${TMPDIR:-/tmp}/nightstream-foundation.XXXXXX")"
trap 'rm -rf "$parity_dir"' EXIT

python3 - "$parity_dir" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

with ZipFile("crates/neo-math/tests/fixtures/lean-foundation.zip") as archive:
    for name in ("field.jsonl", "extension.jsonl", "bar.jsonl", "split.jsonl"):
        Path(sys.argv[1], name).write_bytes(archive.read(name))
PY

# Project policy: at most 300 s per Rust test.
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir" |
  timeout -s KILL 300 env RUSTC_WRAPPER= cargo test -p neo-math --release \
    --test field_lean_parity -- --ignored --exact active_lean_arithmetic_matches_runtime
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/bar.jsonl" |
  timeout -s KILL 300 env RUSTC_WRAPPER= cargo test -p neo-math --release \
    --test phi81_bar_lean_parity -- --ignored --exact active_lean_bar_matches_runtime
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/split.jsonl" |
  timeout -s KILL 300 env RUSTC_WRAPPER= cargo test -p neo-reductions --release \
    --test signed_binary_lean_parity -- --ignored --exact active_lean_signed_binary_matches_runtime
