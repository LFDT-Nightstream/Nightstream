#!/usr/bin/env bash
# Validation enforces the project cap and reports every requested target.
#   validate.sh static            boundary checks here and in ../compression-lean (no Lean)
#   validate.sh build [target...] lake build (default: the production library)
#   validate.sh axioms            lake build NightstreamFPrimeTests
#   validate.sh executables       lake build of every lean_exe target
#   validate.sh stage1-axioms     focused Stage 1 and matrix axiom audits
#   validate.sh file <path.lean>  lake env lean <path>, warnings fail
#   validate.sh lean-executable <built-Lean-executable> [arguments...]
#   validate.sh pi-rlc-sampler-parity <path>
#   validate.sh ajtai-setup-v1-parity <path>
#   validate.sh foundation-parity <directory>
#   validate.sh compression       boundary checks, libraries and executables of
#                                 ../compression-lean, which imports this package
#   validate.sh all               static, both libraries, executables, census
#                                 and compression
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
  bash scripts/check-boundaries.sh ../compression-lean
}

compression() {
  # A missing cache, or one at another Mathlib revision than the manifest,
  # makes Lake fetch Mathlib and rebuild it under the cap.
  local want
  want="$(awk '/"rev":/ { rev = $2 } /"name": "mathlib"/ { gsub(/[",]/, "", rev); print rev }' \
    ../compression-lean/lake-manifest.json)"
  if [[ -z "$want" || "$(git -C ../compression-lean/.lake/packages/mathlib rev-parse HEAD 2>/dev/null)" != "$want" ]]; then
    echo "../compression-lean has no Mathlib cache at revision ${want:-?} of its lake-manifest.json. Replace it with the F′ cache first:" >&2
    echo "  (cd ../compression-lean && rm -rf .lake/packages && mkdir -p .lake && cp -cR ../nightstream-fprime/.lake/packages .lake/)" >&2
    return 2
  fi
  # No target names: Lake builds the package's own defaultTargets.
  # Lake does not track the files that the ports gate and the census read
  # (Lean and Rust sources, STATUS.md's tests, SPEC.md); rerun both.
  (cd ../compression-lean &&
    capped lake build --wfail &&
    capped lake build --wfail $(executables) &&
    lean_file CompressionLeanTests/Ports.lean &&
    lean_file CompressionLeanTests/SpecCensus.lean)
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
  foundation-parity)
    if (( $# != 2 )); then echo "usage: validate.sh foundation-parity <directory>" >&2; exit 2; fi
    capped lake exe emitFoundationParity -- "$2"
    ;;
  ajtai-setup-v1-parity)
    if (( $# != 2 )); then echo "usage: validate.sh ajtai-setup-v1-parity <path>" >&2; exit 2; fi
    capped lake exe emitAjtaiSetupV1Parity -- "$2"
    ;;
  compression)
    bash scripts/check-boundaries.sh ../compression-lean
    compression
    ;;
  all)
    boundaries
    capped lake build --wfail NightstreamFPrime NightstreamFPrimeTests
    capped lake build --wfail $(executables)
    lean_file tests/EndpointCensus.lean
    compression
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
