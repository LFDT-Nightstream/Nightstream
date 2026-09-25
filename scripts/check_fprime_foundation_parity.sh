#!/usr/bin/env bash
# Fresh active-Lean output followed by bounded Rust primitive comparisons.
set -euo pipefail
parity_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$parity_root"
mkdir -p formal/nightstream-fprime/.lake
parity_dir="$(mktemp -d "$parity_root/formal/nightstream-fprime/.lake/foundation-parity.XXXXXX")"
trap 'rm -rf "$parity_dir"' EXIT

# Project policy: at most 1,500 s per Lean command and 300 s per Rust test.
timeout -s KILL 1500 formal/nightstream-fprime/scripts/validate.sh foundation-parity "$parity_dir"
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/bar.jsonl" |
  timeout -s KILL 300 env RUSTC_WRAPPER= cargo test -p neo-math --release \
    --test phi81_bar_lean_parity -- --ignored --exact active_lean_bar_matches_runtime
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/split.jsonl" |
  timeout -s KILL 300 env RUSTC_WRAPPER= cargo test -p neo-reductions --release \
    --test signed_binary_lean_parity -- --ignored --exact active_lean_signed_binary_matches_runtime
