#!/usr/bin/env bash
# Validation enforces the project cap and reports every requested target.
#   validate.sh static            boundary checks (no Lean)
#   validate.sh build [target...] lake build (default: the production library)
#   validate.sh axioms            lake build NightstreamFPrimeTests
#   validate.sh executables       lake build of every lean_exe target
#   validate.sh stage1-axioms     focused Stage 1 and matrix axiom audits
#   validate.sh file <path.lean>  lake env lean <path>, warnings fail
#   validate.sh lean-executable <built-Lean-executable> [arguments...]
#   validate.sh pi-rlc-sampler-parity <path>
#   validate.sh compression-parity <path>
#   validate.sh fold-gate-parity <path>
#   validate.sh ajtai-setup-v1-parity <path>
#   validate.sh foundation-parity <directory>
#   validate.sh all               static, both libraries, executables and census
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CAP="${LEAN_TIMEOUT_SECONDS:-1500}"
if [[ ! "$CAP" =~ ^[0-9]+$ ]] || (( CAP < 1 || CAP > 1500 )); then
  echo "LEAN_TIMEOUT_SECONDS must be between 1 and 1500" >&2; exit 2
fi
LEAN_NUM_THREADS="${LEAN_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN)}"
if [[ ! "$LEAN_NUM_THREADS" =~ ^[0-9]+$ ]] || (( LEAN_NUM_THREADS < 1 )); then
  echo "LEAN_NUM_THREADS must be a positive integer" >&2; exit 2
fi
export LEAN_NUM_THREADS
echo "[parallel] LEAN_NUM_THREADS=${LEAN_NUM_THREADS}"

capped() {
  local start=$SECONDS rc=0
  echo "[bounded ${CAP}s] $*"
  timeout --signal=KILL "$CAP" "$@" || rc=$?
  echo "[bounded] exit=$rc elapsed=$((SECONDS - start))s"
  if (( rc == 124 || rc == 137 )); then echo "[bounded] TIMEOUT is a failed gate" >&2; fi
  return $rc
}

lean_file() {
  capped lake env lean "-j${LEAN_NUM_THREADS}" \
    -DautoImplicit=false -DrelaxedAutoImplicit=false -DwarningAsError=true "$1"
}

boundaries() {
  bash scripts/check-boundaries.sh
}

executables() {
  awk '/^\[\[lean_exe\]\]/ { exe = 1; next } /^\[\[/ { exe = 0 }
    exe && /^name *=/ { gsub(/"/, "", $3); print $3 }' lakefile.toml
}

phase="${1:-all}"
case "$phase" in
  static) boundaries ;;
  lean-executable)
    if (( $# < 2 )); then echo "usage: validate.sh lean-executable <built-Lean-executable> [arguments...]" >&2; exit 2; fi
    shift
    capped "$@"
    ;;
  build)
    shift
    if (( $# == 0 )); then set -- NightstreamFPrime; fi
    capped lake build --wfail "$@"
    ;;
  axioms)
    # Lake does not track the documents that the census reads; rerun it.
    capped lake build --wfail NightstreamFPrimeTests && lean_file tests/EndpointCensus.lean
    ;;
  executables)
    capped lake build --wfail $(executables)
    ;;
  stage1-axioms)
    for audit in \
      tests/AxiomsAjtaiSetupV1.lean \
      tests/AxiomsPiCCSClosure.lean \
      tests/AxiomsStage1Accumulator.lean \
      tests/AxiomsStage1Application.lean \
      tests/AxiomsStage1Assembler.lean \
      tests/AxiomsStage1PiDEC.lean \
      tests/AxiomsStage1PiRLCExport.lean \
      tests/AxiomsStage1Security.lean \
      tests/AxiomsProductionMatrixPlan.lean
    do
      lean_file "$audit"
    done
    ;;
  file)   lean_file "$2" ;;
  pi-rlc-sampler-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-rlc-sampler-parity <path>" >&2; exit 2; fi
    capped lake exe emitPiRlcSamplerParity -- "$2"
    ;;
  compression-parity)
    if (( $# != 2 )); then echo "usage: validate.sh compression-parity <path>" >&2; exit 2; fi
    capped lake exe emitCompressionParity -- "$2"
    ;;
  fold-gate-parity)
    if (( $# != 2 )); then echo "usage: validate.sh fold-gate-parity <path>" >&2; exit 2; fi
    capped lake exe emitFoldGateParity -- "$2"
    ;;
  foundation-parity)
    if (( $# != 2 )); then echo "usage: validate.sh foundation-parity <directory>" >&2; exit 2; fi
    capped lake exe emitFoundationParity -- "$2"
    ;;
  ajtai-setup-v1-parity)
    if (( $# != 2 )); then echo "usage: validate.sh ajtai-setup-v1-parity <path>" >&2; exit 2; fi
    capped lake exe emitAjtaiSetupV1Parity -- "$2"
    ;;
  all)
    boundaries
    capped lake build --wfail NightstreamFPrime NightstreamFPrimeTests
    capped lake build --wfail $(executables)
    lean_file tests/EndpointCensus.lean
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
