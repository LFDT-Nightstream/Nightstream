#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_DIR="$ROOT/formal/nightstream-lean"
SOURCE_PATHS=(
  "$FORMAL_DIR/Nightstream"
  "$FORMAL_DIR/tests"
  "$FORMAL_DIR/Main.lean"
  "$FORMAL_DIR/Nightstream.lean"
)

cd "$ROOT"

echo "[audit] scanning active Nightstream Lean sources for forbidden trusted holes"

if rg -n --glob '*.lean' --glob '!.lake/**' \
  '\b(sorry|axiom|admit|postulate|unsafe)\b' "${SOURCE_PATHS[@]}"; then
  echo "[audit] forbidden token found in formal Lean sources" >&2
  exit 1
fi

echo "[audit] active Nightstream Lean sources are free of sorry/axiom/admit/postulate/unsafe"
