#!/usr/bin/env bash
set -euo pipefail
binding_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$binding_root"
mkdir -p formal/nightstream-fprime/.lake
binding_dir="$(mktemp -d "$binding_root/formal/nightstream-fprime/.lake/wasm-binding.XXXXXX")"
trap 'rm -rf "$binding_dir"' EXIT
lean_cap_seconds=1500

cargo run --release -p neo-wasm --bin export_wasm_rocq -- --binding "$binding_dir"
(cd "$binding_dir" && sha256sum -c "$binding_root/formal/nightstream-fprime/scripts/wasm-binding.sha256")
timeout -s KILL "$lean_cap_seconds" formal/nightstream-fprime/scripts/validate.sh wasm-binding \
  "$binding_dir"/wasm-binding-{micro,small,medium,large}-v1.json
for mutated in $(python3 -B scripts/wasm_binding_mutations.py "$binding_dir"); do
  if formal/nightstream-fprime/scripts/validate.sh wasm-binding "$mutated"; then
    echo "accepted a mutated export: $mutated" >&2
    exit 1
  fi
done
