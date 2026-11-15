#!/usr/bin/env bash
# Regenerates every Lean-emitted artifact and fixture, then the zstd copies of the embedded
# packages and the SHA256SUMS manifests. Run it after a `lake build` of the executables
# (`formal/nightstream-fprime/scripts/validate.sh executables`), never during one.
#
# The emitter runs are independent: each reads only the built Lean executable and writes
# only its own output files, which no other run reads or writes. So they run in parallel,
# as many at a time as the host has cores and memory for (about 4 GiB each). Each run keeps
# its own 1,500 s cap through `validate.sh lean-executable`. The script then checks that
# every output file was written by this run.
#
#   scripts/regenerate_artifacts.sh [log directory]
set -uo pipefail
export PATH="$HOME/.elan/bin:/opt/homebrew/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${1:-$(mktemp -d)}"
mkdir -p "$LOGS"
cd "$ROOT/formal/nightstream-fprime"

cores=$(getconf _NPROCESSORS_ONLN)
if memory=$(sysctl -n hw.memsize 2>/dev/null); then :; else
  memory=$(( $(awk '/^MemTotal:/ { print $2 }' /proc/meminfo) * 1024 ))
fi
jobs=$(( memory / (4 << 30) ))
(( jobs > cores )) && jobs=$cores
(( jobs < 1 )) && jobs=1

B=.lake/build/bin
A=artifacts
F=../../crates/nightstream/tests/fixtures
# One line per run: the emitter and its arguments, then `|` and the files that it writes.
runs=(
  "$B/emitBoundedDepth|$A/nightstream-fprime-bounded-depth-light-v1.json $A/nightstream-fprime-bounded-depth-refresh-v1.json $A/nightstream-fprime-bounded-depth-parity-v1.json"
  "$B/emitBoundedDepth --program nebula-v1|$A/nightstream-fprime-bounded-depth-nebula-v1-light-v1.json $A/nightstream-fprime-bounded-depth-nebula-v1-refresh-v1.json"
  "$B/emitBoundedDepth --program nebula-n1|$A/nightstream-fprime-bounded-depth-nebula-n1-light-v1.json $A/nightstream-fprime-bounded-depth-nebula-n1-refresh-v1.json"
)
for mode in rows start terminal; do
  for plan in v1 n1; do
    runs+=("$B/emitBoundedDepth --application-$mode nebula-$plan $F/nebula-memory-$plan-$mode.json|$F/nebula-memory-$plan-$mode.json")
  done
done
runs+=(
  "$B/emitBoundedDepth --application-start nebula-v1-other $F/nebula-memory-v1-other-start.json|$F/nebula-memory-v1-other-start.json"
  "$B/emitBoundedDepth --application-digest poseidon2-hash-chain-v1 $F/poseidon2-hash-chain-v1-digest.json|$F/poseidon2-hash-chain-v1-digest.json"
)
for plan in v1 n1 r1; do
  runs+=("$B/emitBoundedDepth --application-digest nebula-$plan $F/nebula-memory-$plan-digest.json|$F/nebula-memory-$plan-digest.json")
done
for pair in \
  "emitSharedFormulas:../../crates/nightstream-fprime/artifacts/shared-formulas-v1.json" \
  "emitPiCCSTestErrorParity:$A/nightstream-fprime-stage1-piccs-test-error-v1.json" \
  "emitPiRlcSamplerParity:$A/nightstream-fprime-stage1-pi-rlc-sampler-v1.json" \
  "emitAjtaiSetupV1Parity:$A/nightstream-fprime-ajtai-setup-v1-parity.json" \
  "emitPoseidon2Vectors:$A/nightstream-fprime-poseidon2-vectors-v1.json" \
  "emitFoldGateParity:$A/nightstream-fprime-fold-gate-parity-v1.json" \
  "emitNebulaGateParity:$A/nightstream-fprime-nebula-gate-parity-v1.json" \
  "emitEncodingParity:$A/nightstream-fprime-encoding-parity-v1.json" \
  "emitMatrixPassParity:$A/nightstream-fprime-matrix-pass-parity-v1.json" \
  "emitCompressionParity:$A/nightstream-fprime-compression-parity-v1.json"; do
  runs+=("$B/${pair%%:*} ${pair#*:}|${pair#*:}")
done

start=$(mktemp)
emit() { # <index> <run>
  local command=${2%%|*}
  # shellcheck disable=SC2086
  if bash scripts/validate.sh lean-executable $command > "$LOGS/run-$1.log" 2>&1; then
    echo "ok: $command"
  else
    echo "FAILED: $command (log $LOGS/run-$1.log)"
  fi
}
for index in "${!runs[@]}"; do
  while (( $(jobs -rp | wc -l) >= jobs )); do sleep 1; done
  emit "$index" "${runs[$index]}" &
done
wait

status=0
for run in "${runs[@]}"; do
  for output in ${run#*|}; do
    if [[ ! "$output" -nt "$start" ]]; then echo "NOT WRITTEN: $output"; status=1; fi
  done
done
if grep -q "TIMEOUT\|exit=[1-9]" "$LOGS"/run-*.log; then status=1; fi
rm -f "$start"

cd "$ROOT/crates/nightstream/artifacts"
for circuit in light refresh; do
  zstd -19 -q -f -o "nightstream-fprime-bounded-depth-$circuit-v1.json.zst" \
    < "../../../formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-$circuit-v1.json" || status=1
done
cd "$ROOT/formal/nightstream-fprime"
python3 -B scripts/check_artifacts.py --write || status=1
echo "${#runs[@]} runs, $jobs at a time, logs in $LOGS, status $status"
exit $status
