#!/usr/bin/env bash
# Make the Akita schedule files of crates/neo-compress/artifacts with the
# planner of Akita (crates/akita-planner), at the Akita revision of
# crates/neo-compress/Cargo.toml.
#
#   scripts/akita_schedules.sh <Akita checkout at that revision>
#
# Each file holds the rows of the family fp64_dense_recursive that open the
# key polynomial, W and V in one proof, for one size of the key polynomial.
# The planner is deterministic. The script prints the SHA-256 of each file,
# which crates/neo-compress/tests/unit/akita.rs pins. One run takes about
# six minutes.
set -euo pipefail

akita="${1:?usage: $0 <Akita checkout>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts="$root/crates/neo-compress/artifacts"
revision="$(sed -n 's/^akita-pcs = .*rev = "\([0-9a-f]*\)".*/\1/p' "$root/crates/neo-compress/Cargo.toml")"
if [ "$(git -C "$akita" rev-parse HEAD)" != "$revision" ]; then
  echo "the Akita checkout is not at revision $revision" >&2
  exit 1
fi
out="$(mktemp -d)"

# plan <file> <variables of V> <variables of the key polynomial> <variables of W>
plan() {
  (cd "$akita" && cargo run --release -p akita-planner --features catalog-gen \
    --bin gen_schedule_artifacts -- "$out/$1" \
    --final-group "fp64_dense_recursive:$2:1" \
    --precommitted-group "fp64_dense_recursive:$3:1" \
    --precommitted-group "fp64_dense_recursive:$4:1")
  cp "$out/$1/fp64_dense_recursive.aks" "$artifacts/$1.aks"
}

plan key20 16 20 16
plan key30 28 30 26..=27

if command -v sha256sum >/dev/null; then
  (cd "$artifacts" && sha256sum key20.aks key30.aks)
else
  (cd "$artifacts" && shasum -a 256 key20.aks key30.aks)
fi
