#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
R1CS="$ROOT/Nightstream/Implementation/R1CS"
failed=0

while IFS= read -r -d '' file; do
  relative="${file#"$ROOT/"}"
  generated=0
  if rg -qi -m 1 '^(/-! +generated|generated file)' "$file"; then
    generated=1
  fi

  if [[ "$file" == */Artifacts/*/Generated/* ]]; then
    if (( generated == 0 )); then
      echo "$relative: generated-directory module lacks a generated header" >&2
      failed=1
    fi
  elif (( generated != 0 )); then
    echo "$relative: generated module is outside Artifacts/<owner>/Generated" >&2
    failed=1
  fi
done < <(find "$R1CS" -type f -name '*.lean' -print0)

raw_imports="$(
  rg -n '^import Nightstream\.Implementation\.R1CS\.Artifacts\..*\.Generated\.' \
    "$ROOT/Nightstream/SuperNeo" \
    "$ROOT/Nightstream/HyperNova" \
    "$ROOT/Nightstream/Protocol" \
    "$ROOT/Nightstream/Assurance" \
    "$ROOT/Nightstream/Implementation/Encoding" \
    "$ROOT/Nightstream/Implementation/FPrime" \
    "$ROOT/Nightstream/Implementation/Rust" \
    "$ROOT/Nightstream/Implementation.lean" \
    --glob '*.lean' || true
)"
if [[ -n "$raw_imports" ]]; then
  echo "$raw_imports" >&2
  echo "generated artifacts must cross the R1CS boundary through stable modules" >&2
  failed=1
fi

stale_rust_paths="$(
  rg --pcre2 -n \
    'Nightstream/Implementation/R1CS/(?!Artifacts/|Core/|Ownership/|Correspondence/)|import Nightstream\.Implementation\.R1CS\.(?!Artifacts\.|Core\.|Ownership\.|Correspondence\.)' \
    "$REPO_ROOT/crates/neo-fold-clean" \
    --glob '*.rs' \
    --glob '!target/**' || true
)"
if [[ -n "$stale_rust_paths" ]]; then
  echo "$stale_rust_paths" >&2
  echo "Rust artifact registries still reference the retired flat R1CS layout" >&2
  failed=1
fi

if (( failed != 0 )); then
  echo "[generated-layout] generated artifact quarantine check failed" >&2
  exit 1
fi

echo "[generated-layout] generated artifact quarantine check passed"
