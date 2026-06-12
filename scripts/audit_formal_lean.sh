#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_DIR="$ROOT/formal/superneo-lean"

cd "$ROOT"

echo "[audit] scanning Lean sources for forbidden trusted holes"

if grep -rwEn --include='*.lean' --exclude-dir='.lake' '(sorry|axiom|admit|postulate|unsafe)' "$FORMAL_DIR"; then
  echo "[audit] forbidden token found in formal Lean sources" >&2
  exit 1
fi

echo "[audit] formal Lean sources are free of sorry/axiom/admit/postulate/unsafe"
