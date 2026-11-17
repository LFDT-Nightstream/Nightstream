#!/usr/bin/env bash
set -euo pipefail
parity_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$parity_root"
mkdir -p formal/nightstream-fprime/.lake
parity_dir="$(mktemp -d "$parity_root/formal/nightstream-fprime/.lake/foundation-parity.XXXXXX")"
trap 'rm -rf "$parity_dir"' EXIT
lean_cap_seconds=1500
rust_test_cap_seconds=300

timeout -s KILL "$lean_cap_seconds" formal/nightstream-fprime/scripts/validate.sh foundation-parity "$parity_dir"
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/bar.jsonl" |
  timeout -s KILL "$rust_test_cap_seconds" env RUSTC_WRAPPER= cargo test -p neo-math --release \
    --test bar_lean_parity -- --ignored --exact active_lean_bar_matches_runtime
python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$parity_dir/split.jsonl" |
  timeout -s KILL "$rust_test_cap_seconds" env RUSTC_WRAPPER= cargo test -p neo-reductions --release \
    --test signed_binary_lean_parity -- --ignored --exact active_lean_signed_binary_matches_runtime
