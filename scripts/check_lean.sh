#!/usr/bin/env bash
# Local Lean assurance, identity and fresh primitive-result checks.
set -euo pipefail
check_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$check_root"

# Project policy: at most 1,500 s per Lean command.
formal/nightstream-fprime/scripts/validate.sh all
formal/nightstream-fprime/scripts/validate.sh identity

parity_dir="$(mktemp -d "${TMPDIR:-/tmp}/nightstream-lean-foundation.XXXXXX")"
trap 'rm -rf "$parity_dir"' EXIT
formal/nightstream-fprime/scripts/validate.sh foundation-parity "$parity_dir"
python3 - "$parity_dir" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

with ZipFile("crates/neo-math/tests/fixtures/lean-foundation.zip") as archive:
    for name in ("field.jsonl", "extension.jsonl", "bar.jsonl", "split.jsonl"):
        if Path(sys.argv[1], name).read_bytes() != archive.read(name):
            raise SystemExit(f"Fresh Lean output differs from the committed result: {name}")
print("Fresh Lean primitive results match the committed files.")
PY

bash scripts/check_fprime_foundation_parity.sh
echo "Local Lean assurance, identity and primitive parity passed."
