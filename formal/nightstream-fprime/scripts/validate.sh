#!/usr/bin/env bash
# Bounded validation. Every Lean invocation runs under the 1500 s hard cap.
#   validate.sh static            boundary checks only (no Lean)
#   validate.sh build [target]    lake build (default: the two libraries)
#   validate.sh axioms            lake build NightstreamFPrimeTests
#   validate.sh file <path.lean>  lake env lean <path>
#   validate.sh emit <path>       lake exe emit -- <path>
#   validate.sh emit-expanded <path>
#   validate.sh pilot-parity <path>
#   validate.sh pi-ccs-parity <path>
#   validate.sh pi-rlc-sampler-parity <path>
#   validate.sh pi-rlc-parity <path>
#   validate.sh pi-dec-parity <path>
#   validate.sh all
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
  echo "[bounded ${CAP}s] $*"
  local start=$SECONDS
  # -k kills hard 10 s after the cap; exit 124 marks a timeout.
  timeout -k 10 "$CAP" "$@"
  local rc=$?
  echo "[bounded] exit=$rc elapsed=$((SECONDS - start))s"
  if (( rc == 124 )); then echo "[bounded] TIMEOUT is a failed gate" >&2; fi
  return $rc
}

phase="${1:-all}"
case "$phase" in
  static) bash scripts/check-boundaries.sh ;;
  build)  capped lake build "${2:-NightstreamFPrime}" ;;
  axioms) capped lake build NightstreamFPrimeTests ;;
  file)   capped lake env lean "-j${LEAN_NUM_THREADS}" -DautoImplicit=false -DrelaxedAutoImplicit=false "$2" ;;
  emit)
    if (( $# != 2 )); then echo "usage: validate.sh emit <path>" >&2; exit 2; fi
    capped lake exe emit -- "$2"
    ;;
  emit-expanded)
    if (( $# != 2 )); then echo "usage: validate.sh emit-expanded <path>" >&2; exit 2; fi
    capped lake exe emit -- --expanded "$2"
    ;;
  pilot-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pilot-parity <path>" >&2; exit 2; fi
    capped lake exe emitPilotParity -- "$2"
    ;;
  pi-ccs-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-ccs-parity <path>" >&2; exit 2; fi
    capped lake exe emitPiCCSParity -- "$2"
    ;;
  pi-rlc-sampler-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-rlc-sampler-parity <path>" >&2; exit 2; fi
    capped lake exe emitPiRlcSamplerParity -- "$2"
    ;;
  pi-rlc-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-rlc-parity <path>" >&2; exit 2; fi
    capped lake exe emitPiRLCParity -- "$2"
    ;;
  pi-dec-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-dec-parity <path>" >&2; exit 2; fi
    capped lake exe emitPiDECParity -- "$2"
    ;;
  all)
    bash scripts/check-boundaries.sh
    capped lake build NightstreamFPrime
    capped lake build NightstreamFPrimeTests
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
