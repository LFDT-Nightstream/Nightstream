#!/usr/bin/env bash
# Validation defaults to the project cap; an owner-authorized zero disables it.
#   validate.sh static            boundary checks only (no Lean)
#   validate.sh build [target...] lake build (default: the production library)
#   validate.sh axioms            lake build NightstreamFPrimeTests
#   validate.sh identity          recompute canonical binding and compare pins
#   validate.sh stage1-axioms     focused Stage 1 and matrix axiom audits
#   validate.sh file <path.lean>  lake env lean <path>
#   validate.sh emit <path>       lake exe emit -- <path>
#   validate.sh emit-expanded <path>
#   validate.sh emit-poseidon2-hash-chain-v1 <path>
#   validate.sh emit-poseidon2-hash-chain-v1-expanded <path>
#   validate.sh pilot-parity <vk0> <vk1> <vk2> <vk3> <path>
#   validate.sh base-step-fixture <vk0> <vk1> <vk2> <vk3> <path>
#   validate.sh recursive-step-fixture <context[4]> <PiCCS-input> <child-running> [<prior-state-message>] <path>
#   validate.sh pi-ccs-parity <vk0> <vk1> <vk2> <vk3> <path>
#   validate.sh pi-ccs-input-check <input-json-path> <output-path>
#   validate.sh pi-rlc-input-check <package[4]> <PiCCS-input> <output-path>
#   validate.sh pi-rlc-witness-replay <PiCCS-input> <source-capture> <output> <start-block> <end-block>
#   validate.sh pi-dec-witness-replay <Lean-parent-range> <output> <start-block> <end-block>
#   validate.sh pi-dec-witness-replay-boundaries <new-results-directory>
#   validate.sh pi-dec-commitment-block <Lean-parent-range> <output>
#   validate.sh pi-dec-commitment-replay <Lean-parent-range> <output> <start-block> <end-block>
#   validate.sh pi-dec-commitment-merge <output> <Lean-range>...
#   validate.sh pi-dec-commitment-merge-boundaries <Lean-range> <new-results-directory>
#   validate.sh pi-dec-evaluation-block [<C-input>] <Lean-parent-range> <output>
#   validate.sh pi-dec-evaluation-replay pad <C-input> <Lean-parent-range> <output> <start> <end>
#   validate.sh pi-dec-pad-merge <output> <range>...
#   validate.sh pi-dec-pad-merge-boundaries <new-results-directory>
#   validate.sh pi-dec-matrix-rows <new-output.jsonl> <first-block> <last-block-exclusive>
#   validate.sh pi-dec-matrix-range <C-input> <new-output> <block> <first-local-row> <last-exclusive> <Lean-parent-ranges>...
#   validate.sh pi-dec-matrix-ranges <C-input> <new-output block first last>... -- <Lean-parent-ranges>...
#   validate.sh lean-executable <built-Lean-executable> [arguments...]
#   validate.sh pi-dec-matrix-merge <C-input> <complete-Pad> <new-output> <matrix-ranges>...
#   validate.sh pi-dec-matrix-merge-boundaries <valid-C-input> <complete-Pad> <new-results-directory>
#   validate.sh pi-dec-parent-boundaries <valid-C-input> <new-results-directory>
#   validate.sh pi-dec-input-check <package[4]> <PiCCS-input> <children> <output-path>
#   validate.sh pi-dec-mutations <package[4]> <PiCCS-input> <children> <bad-PiCCS-input> <mutations-dir>
#   validate.sh pi-ccs-ownership-audit <id0> <id1> <id2> <id3> <path>
#   validate.sh pi-rlc-sampler-parity <path>
#   validate.sh pi-rlc-parity <context[4]> <package[4]> <path>
#   validate.sh pi-dec-parity <context[4]> <package[4]> <path>
#   validate.sh foundation-parity <directory>
#   validate.sh ajtai-setup-v1-parity <path>
#   validate.sh ajtai-sparse-commitment-v1-parity <path>
#   validate.sh poseidon2-hash-chain-v1-parity <context[4]> <path>
#   validate.sh poseidon2-hash-chain-v1-canonical-binding <path>
#   validate.sh poseidon2-hash-chain-v1-binding-parity <id[4]> <relation[4]> <application[4]> <nifs[4]> <commitment[4]> <path>
#   validate.sh per-application-reference <path>
#   validate.sh per-application-streamed <path>
#   validate.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CAP="${LEAN_TIMEOUT_SECONDS:-1500}"
if [[ ! "$CAP" =~ ^[0-9]+$ ]] || (( CAP < 0 || CAP > 1500 )); then
  echo "LEAN_TIMEOUT_SECONDS must be 0 (owner-authorized no timeout) or between 1 and 1500" >&2; exit 2
fi
LEAN_NUM_THREADS="${LEAN_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN)}"
if [[ ! "$LEAN_NUM_THREADS" =~ ^[0-9]+$ ]] || (( LEAN_NUM_THREADS < 1 )); then
  echo "LEAN_NUM_THREADS must be a positive integer" >&2; exit 2
fi
export LEAN_NUM_THREADS
echo "[parallel] LEAN_NUM_THREADS=${LEAN_NUM_THREADS}"

capped() {
  local start=$SECONDS
  if (( CAP == 0 )); then
    echo "[no timeout] $*"
    "$@"
  else
    echo "[bounded ${CAP}s] $*"
    # -k kills hard 10 s after the cap; exit 124 marks a timeout.
    timeout -k 10 "$CAP" "$@"
  fi
  local rc=$?
  if (( CAP == 0 )); then
    echo "[no timeout] exit=$rc elapsed=$((SECONDS - start))s"
  else
    echo "[bounded] exit=$rc elapsed=$((SECONDS - start))s"
  fi
  if (( rc == 124 )); then echo "[bounded] TIMEOUT is a failed gate" >&2; fi
  return $rc
}

lean_file() {
  capped lake env lean "-j${LEAN_NUM_THREADS}" \
    -DautoImplicit=false -DrelaxedAutoImplicit=false "$1"
}

phase="${1:-all}"
case "$phase" in
  static) bash scripts/check-boundaries.sh ;;
  lean-executable)
    if (( $# < 2 )); then echo "usage: validate.sh lean-executable <built-Lean-executable> [arguments...]" >&2; exit 2; fi
    shift
    capped "$@"
    ;;
  build)
    shift
    if (( $# == 0 )); then set -- NightstreamFPrime; fi
    capped lake build "$@"
    ;;
  axioms) capped lake build NightstreamFPrimeTests ;;
  pi-ccs-first-round)
    if (( $# != 6 )); then echo "usage: validate.sh pi-ccs-first-round <public-input> <original-sources> <output> <first-pair> <end-pair>" >&2; exit 2; fi
    shift
    capped lake exe replayPiCCSFirstRound -- "$@"
    ;;
  pi-rlc-witness-replay)
    if (( $# != 6 )); then echo "usage: validate.sh pi-rlc-witness-replay <C-input> <source-capture> <output> <start-block> <end-block>" >&2; exit 2; fi
    shift
    capped lake exe replayPiRLCWitness -- "$@"
    ;;
  pi-dec-witness-replay)
    if (( $# != 5 )); then echo "usage: validate.sh pi-dec-witness-replay <Lean-parent-range> <output> <start-block> <end-block>" >&2; exit 2; fi
    shift
    capped lake exe replayPiDECWitness -- "$@"
    ;;
  pi-dec-witness-replay-boundaries)
    if (( $# != 2 )); then echo "usage: validate.sh pi-dec-witness-replay-boundaries <new-results-directory>" >&2; exit 2; fi
    capped python3 -B tests/pi_dec_witness_replay.py "$2"
    ;;
  pi-dec-commitment-block)
    if (( $# != 3 )); then echo "usage: validate.sh pi-dec-commitment-block <Lean-parent-range> <output>" >&2; exit 2; fi
    capped lake exe measurePiDECCommitmentBlock -- "$2" "$3"
    ;;
  pi-dec-commitment-replay)
    if (( $# != 5 )); then echo "usage: validate.sh pi-dec-commitment-replay <Lean-parent-range> <output> <start-block> <end-block>" >&2; exit 2; fi
    shift
    capped lake exe replayPiDECCommitment -- "$@"
    ;;
  pi-dec-commitment-merge)
    if (( $# < 3 )); then echo "usage: validate.sh pi-dec-commitment-merge <output> <Lean-range>..." >&2; exit 2; fi
    shift
    capped lake exe replayPiDECCommitment -- merge "$@"
    ;;
  pi-dec-commitment-merge-boundaries)
    if (( $# != 3 )); then echo "usage: validate.sh pi-dec-commitment-merge-boundaries <Lean-range> <new-results-directory>" >&2; exit 2; fi
    timeout -k 10 300 python3 -B tests/pi_dec_commitment_merge.py "$2" "$3"
    ;;
  pi-dec-evaluation-replay)
    if (( $# != 7 )); then echo "usage: validate.sh pi-dec-evaluation-replay pad <C-input> <Lean-parent-range> <output> <start> <end>" >&2; exit 2; fi
    shift
    capped lake exe replayPiDECEvaluation -- "$@"
    ;;
  pi-dec-pad-merge)
    if (( $# < 3 )); then echo "usage: validate.sh pi-dec-pad-merge <output> <range>..." >&2; exit 2; fi
    shift
    capped lake exe replayPiDECEvaluation -- merge-pad "$@"
    ;;
  pi-dec-pad-merge-boundaries)
    if (( $# != 2 )); then echo "usage: validate.sh pi-dec-pad-merge-boundaries <new-results-directory>" >&2; exit 2; fi
    timeout -k 10 300 python3 -B tests/pi_dec_pad_merge.py .lake/build/bin/replayPiDECEvaluation "$2"
    ;;
  pi-dec-matrix-rows)
    if (( $# != 4 )); then echo "usage: validate.sh pi-dec-matrix-rows <new-output.jsonl> <first-block> <last-block-exclusive>" >&2; exit 2; fi
    capped lake exe measurePiDECMatrixRows -- "$2" "$3" "$4"
    ;;
  pi-dec-matrix-range)
    if (( $# < 7 )); then echo "usage: validate.sh pi-dec-matrix-range <C-input> <new-output> <block> <first-local-row> <last-exclusive> <Lean-parent-ranges>..." >&2; exit 2; fi
    shift
    capped lake exe replayPiDECMatrix -- "$@"
    ;;
  pi-dec-matrix-ranges)
    if (( $# < 8 )); then echo "usage: validate.sh pi-dec-matrix-ranges <C-input> <new-output block first last>... -- <Lean-parent-ranges>..." >&2; exit 2; fi
    shift
    capped lake exe replayPiDECMatrix -- ranges "$@"
    ;;
  pi-dec-parent-boundaries)
    if (( $# != 3 )); then echo "usage: validate.sh pi-dec-parent-boundaries <valid-C-input> <new-results-directory>" >&2; exit 2; fi
    timeout -k 10 300 python3 -B tests/pi_dec_parent_input.py .lake/build/bin/replayPiDECMatrix "$2" "$3"
    ;;
  pi-dec-matrix-merge)
    if (( $# < 5 )); then echo "usage: validate.sh pi-dec-matrix-merge <C-input> <complete-Pad> <new-output> <matrix-ranges>..." >&2; exit 2; fi
    shift
    capped lake exe mergePiDECMatrix -- "$@"
    ;;
  pi-dec-matrix-merge-boundaries)
    if (( $# != 4 )); then echo "usage: validate.sh pi-dec-matrix-merge-boundaries <valid-C-input> <complete-Pad> <new-results-directory>" >&2; exit 2; fi
    timeout -k 10 300 python3 -B tests/pi_dec_matrix_merge.py .lake/build/bin/mergePiDECMatrix "$2" "$3" "$4"
    ;;
  pi-dec-evaluation-block)
    if (( $# != 3 && $# != 4 )); then echo "usage: validate.sh pi-dec-evaluation-block [<C-input>] <Lean-parent-range> <output>" >&2; exit 2; fi
    shift
    capped lake exe measurePiDECEvaluationBlock -- "$@"
    ;;
  identity)
    if (( $# != 1 )); then echo "usage: validate.sh identity" >&2; exit 2; fi
    identity_output="$(mktemp "${TMPDIR:-/tmp}/nightstream-fprime-identity.XXXXXX.json")"
    trap 'rm -f -- "$identity_output"' EXIT
    capped lake exe emitPoseidon2HashChainV1BindingParity -- "$identity_output"
    timeout --signal=KILL 300s python3 -B scripts/check_identity.py "$identity_output"
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
      tests/AxiomsStage1PiRLCParity.lean \
      tests/AxiomsStage1Security.lean \
      tests/AxiomsProductionMatrixPlan.lean
    do
      lean_file "$audit"
    done
    ;;
  file)   lean_file "$2" ;;
  emit)
    if (( $# != 2 )); then echo "usage: validate.sh emit <path>" >&2; exit 2; fi
    capped lake exe emit -- "$2"
    ;;
  emit-expanded)
    if (( $# != 2 )); then echo "usage: validate.sh emit-expanded <path>" >&2; exit 2; fi
    capped lake exe emit -- --expanded "$2"
    ;;
  emit-poseidon2-hash-chain-v1)
    if (( $# != 2 )); then echo "usage: validate.sh emit-poseidon2-hash-chain-v1 <path>" >&2; exit 2; fi
    capped lake exe emit -- --poseidon2-hash-chain-v1 "$2"
    ;;
  emit-poseidon2-hash-chain-v1-expanded)
    if (( $# != 2 )); then echo "usage: validate.sh emit-poseidon2-hash-chain-v1-expanded <path>" >&2; exit 2; fi
    capped lake exe emit -- --poseidon2-hash-chain-v1-expanded "$2"
    ;;
  pilot-parity)
    if (( $# != 6 )); then
      echo "usage: validate.sh pilot-parity <vk0> <vk1> <vk2> <vk3> <path>" >&2
      exit 2
    fi
    capped lake exe emitPilotParity -- "$2" "$3" "$4" "$5" "$6"
    ;;
  base-step-fixture)
    if (( $# != 6 )); then
      echo "usage: validate.sh base-step-fixture <vk0> <vk1> <vk2> <vk3> <path>" >&2
      exit 2
    fi
    capped lake exe emitBaseStepFixture -- "$2" "$3" "$4" "$5" "$6"
    ;;
  recursive-step-fixture)
    if (( $# != 8 && $# != 9 )); then
      echo "usage: validate.sh recursive-step-fixture <context[4]> <PiCCS-input> <child-running> [<prior-state-message>] <path>" >&2
      exit 2
    fi
    capped lake exe emitRecursiveStepFixture -- "${@:2}"
    ;;
  pi-ccs-parity)
    if (( $# != 6 )); then
      echo "usage: validate.sh pi-ccs-parity <vk0> <vk1> <vk2> <vk3> <path>" >&2
      exit 2
    fi
    capped lake exe emitPiCCSParity -- "$2" "$3" "$4" "$5" "$6"
    ;;
  pi-ccs-ownership-audit)
    if (( $# != 6 )); then
      echo "usage: validate.sh pi-ccs-ownership-audit <id0> <id1> <id2> <id3> <path>" >&2
      exit 2
    fi
    capped lake exe emitPiCCSOwnershipAudit -- "$2" "$3" "$4" "$5" "$6"
    ;;
  pi-ccs-input-check)
    if (( $# != 3 )); then
      echo "usage: validate.sh pi-ccs-input-check <input-json-path> <output-path>" >&2
      exit 2
    fi
    capped lake exe checkPiCCSInput -- "$2" "$3"
    ;;
  pi-rlc-sampler-parity)
    if (( $# != 2 )); then echo "usage: validate.sh pi-rlc-sampler-parity <path>" >&2; exit 2; fi
    capped lake exe emitWideSamplerParity -- "$2"
    ;;
  pi-rlc-input-check)
    if (( $# != 7 )); then
      echo "usage: validate.sh pi-rlc-input-check <package[4]> <PiCCS-input> <output-path>" >&2
      exit 2
    fi
    shift
    capped lake exe checkPiRLCInput -- "$@"
    ;;
  pi-rlc-parity)
    if (( $# != 10 )); then
      echo "usage: validate.sh pi-rlc-parity <context[4]> <package[4]> <path>" >&2
      exit 2
    fi
    shift
    capped lake exe emitPiRLCParity -- "$@"
    ;;
  pi-dec-input-check)
    if (( $# != 8 )); then
      echo "usage: validate.sh pi-dec-input-check <package[4]> <PiCCS-input> <children> <output-path>" >&2
      exit 2
    fi
    shift
    capped lake exe checkPiDECInput -- "$@"
    ;;
  pi-dec-parity)
    if (( $# != 10 )); then
      echo "usage: validate.sh pi-dec-parity <context[4]> <package[4]> <path>" >&2
      exit 2
    fi
    shift
    capped lake exe emitPiDECParity -- "$@"
    ;;
  pi-dec-mutations)
    if (( $# != 9 )); then
      echo "usage: validate.sh pi-dec-mutations <package[4]> <PiCCS-input> <children> <bad-PiCCS-input> <mutations-dir>" >&2
      exit 2
    fi
    shift
    capped lake exe checkPiDECActualMutations "$@"
    ;;
  foundation-parity)
    if (( $# != 2 )); then echo "usage: validate.sh foundation-parity <directory>" >&2; exit 2; fi
    capped lake exe emitFoundationParity -- "$2"
    ;;
  ajtai-setup-v1-parity)
    if (( $# != 2 )); then echo "usage: validate.sh ajtai-setup-v1-parity <path>" >&2; exit 2; fi
    capped lake exe emitAjtaiSetupV1Parity -- "$2"
    ;;
  ajtai-sparse-commitment-v1-parity)
    if (( $# != 2 )); then echo "usage: validate.sh ajtai-sparse-commitment-v1-parity <path>" >&2; exit 2; fi
    capped lake exe emitAjtaiSparseCommitmentV1Parity -- "$2"
    ;;
  poseidon2-hash-chain-v1-parity)
    if (( $# != 6 )); then
      echo "usage: validate.sh poseidon2-hash-chain-v1-parity <context[4]> <path>" >&2
      exit 2
    fi
    shift
    capped lake exe emitPoseidon2HashChainV1Parity -- "$@"
    ;;
  poseidon2-hash-chain-v1-canonical-binding)
    if (( $# != 2 )); then
      echo "usage: validate.sh poseidon2-hash-chain-v1-canonical-binding <path>" >&2
      exit 2
    fi
    capped lake exe emitPoseidon2HashChainV1BindingParity -- "$2"
    ;;
  poseidon2-hash-chain-v1-binding-parity)
    if (( $# != 22 )); then
      echo "usage: validate.sh poseidon2-hash-chain-v1-binding-parity <id0> <id1> <id2> <id3> <relation0> <relation1> <relation2> <relation3> <application0> <application1> <application2> <application3> <nifs0> <nifs1> <nifs2> <nifs3> <commitment0> <commitment1> <commitment2> <commitment3> <path>" >&2
      exit 2
    fi
    capped lake exe emitPoseidon2HashChainV1BindingParity -- \
      "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
      "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" \
      "${18}" "${19}" "${20}" "${21}" "${22}"
    ;;
  per-application-reference)
    if (( $# != 2 )); then echo "usage: validate.sh per-application-reference <path>" >&2; exit 2; fi
    capped lake exe emitPerApplicationReferenceFixture -- "$2"
    ;;
  per-application-streamed)
    if (( $# != 2 )); then echo "usage: validate.sh per-application-streamed <path>" >&2; exit 2; fi
    capped lake exe emitPerApplicationStreamedFixture -- "$2"
    ;;
  all)
    bash scripts/check-boundaries.sh
    capped lake build NightstreamFPrime
    capped lake build NightstreamFPrimeTests
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
