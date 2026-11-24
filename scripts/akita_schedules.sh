#!/usr/bin/env bash
set -euo pipefail

akita="${1:?usage: $0 <Akita checkout>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts="$root/crates/ns-compress/artifacts"
revision="$(sed -n 's/^akita-verifier = .*rev = "\([0-9a-f]*\)".*/\1/p' "$root/crates/ns-compress/Cargo.toml")"
if [ "$(git -C "$akita" rev-parse HEAD)" != "$revision" ]; then
  echo "the Akita checkout is not at revision $revision" >&2
  exit 1
fi
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

git -C "$akita" archive HEAD | tar -x -C "$out"
git -C "$out" apply "$root/scripts/akita-shared-inverse-planner.patch"
git -C "$root" show 2e4818e556e5470f8a7e12af4381481428b7c057:crates/ns-compress/artifacts/key30.aks \
  > "$out/crates/akita-planner/key30-guide.aks"

plan() {
  local file=$1 v_vars=$2 key_vars=$3 w_vars=$4
  (cd "$out" && cargo run --release -p akita-planner --features catalog-gen \
    --bin gen_schedule_artifacts -- "$out/$file" \
    --final-group "fp64_dense_recursive:$v_vars:1" \
    --precommitted-group "fp64_dense_recursive:$key_vars:1" \
    --precommitted-group "fp64_dense_recursive:$w_vars:1")
  cp "$out/$file/fp64_dense_recursive.aks" "$artifacts/$file.aks"
}

plan key20 16 20 16
plan key30 26 30 26..=27

if command -v sha256sum >/dev/null; then
  (cd "$artifacts" && sha256sum key20.aks key30.aks)
else
  (cd "$artifacts" && shasum -a 256 key20.aks key30.aks)
fi
