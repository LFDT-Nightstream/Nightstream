#!/usr/bin/env bash
set -euo pipefail

akita="${1:?usage: $0 <Akita checkout>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts="$root/crates/ns-compress/artifacts"
revision="$(sed -n 's/^akita-pcs = .*rev = "\([0-9a-f]*\)".*/\1/p' "$root/crates/ns-compress/Cargo.toml")"
if [ "$(git -C "$akita" rev-parse HEAD)" != "$revision" ]; then
  echo "the Akita checkout is not at revision $revision" >&2
  exit 1
fi
out="$(mktemp -d)"

plan() {
  local file=$1 v_vars=$2 key_vars=$3 w_vars=$4
  (cd "$akita" && cargo run --release -p akita-planner --features catalog-gen \
    --bin gen_schedule_artifacts -- "$out/$file" \
    --final-group "fp64_dense_recursive:$v_vars:1" \
    --precommitted-group "fp64_dense_recursive:$key_vars:1" \
    --precommitted-group "fp64_dense_recursive:$w_vars:1")
  cp "$out/$file/fp64_dense_recursive.aks" "$artifacts/$file.aks"
}

plan key20 16 20 16
plan key30 28 30 26..=27

if command -v sha256sum >/dev/null; then
  (cd "$artifacts" && sha256sum key20.aks key30.aks)
else
  (cd "$artifacts" && shasum -a 256 key20.aks key30.aks)
fi
