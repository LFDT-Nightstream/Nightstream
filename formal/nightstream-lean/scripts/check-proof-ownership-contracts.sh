#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

roots=(
  formal/nightstream-lean/Nightstream/Protocol/FPrime/ConcretePhi81.lean
  formal/nightstream-lean/Nightstream/Protocol/FPrime/ConcretePhi81
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/Nifs/ConcretePhi81.lean
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/Nifs/ConcretePhi81
  formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/EvaluationHomomorphism.lean
  formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/EvaluationHomomorphism
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/OutputClaims/EvaluationHomomorphism.lean
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/OutputClaims/EvaluationHomomorphism
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint.lean
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/SplitNc.lean
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/SplitNc
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/PiDec.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/PiDec
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/RelabeledCarrier.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/PiRlc.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/NifsPaper/PiRlc
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/PiCcsNc/Authority
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/PiRlcChallenge
  crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling
  crates/neo-fold-clean/src/frontends/f_prime/gadget_native.rs
  crates/neo-fold-clean/src/frontends/f_prime/gadget_native
  crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc
  crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit
)

status=0
while IFS= read -r relative; do
  file="$REPO_ROOT/$relative"
  header="$(sed -n '1,100p' "$file")"
  for marker in 'Owns:' 'Does not own:' 'Emits constraints:'; do
    if ! rg -q -F "$marker" <<< "$header"; then
      echo "[ownership] $relative is missing '$marker' in its first 100 lines" >&2
      status=1
    fi
  done
  if ! rg -q '\|[[:space:]]*---' <<< "$header"; then
    echo "[ownership] $relative is missing a header ownership/equation table" >&2
    status=1
  fi
done < <(cd "$REPO_ROOT" && rg --files -g '*.lean' -g '*.rs' "${roots[@]}" | sort)

# Completion state belongs in the normative property specifications and
# assurance records. A source header may name the guarantee and excluded
# boundary, but an editable Status column must never masquerade as evidence.
while IFS= read -r relative; do
  file="$REPO_ROOT/$relative"
  if sed -n '1,120p' "$file" | rg -q '\|[[:space:]]*Status[[:space:]]*\|'; then
    echo "[ownership] $relative has an editable Status table in its header" >&2
    status=1
  fi
done < <(cd "$REPO_ROOT" && rg --files \
  formal/nightstream-lean/Nightstream -g '*.lean' | sort)

# The active ConcretePhi81 NIFS/F-prime model has one canonical 9/3/6
# BlockLane transcript profile. Keep the superseded flat-domain protocol out
# of this trust path; legacy artifact correspondence remains under
# Implementation/R1CS and must refine into this model explicitly.
canonical_roots=(
  formal/nightstream-lean/Nightstream/SuperNeo/Folding/Nifs/ConcretePhi81
  formal/nightstream-lean/Nightstream/Protocol/FPrime/ConcretePhi81
)
legacy_pattern='\bFlatNcDomain\b|Protocol\.TranscriptAuthority\.Schedule|Protocol\.Certificate|Protocol\.Accepted|Protocol\.derive|\bpiCcsOutputHandoff\b|\boutputPoints\b|\bbetaM\b'
if legacy_hits="$(cd "$REPO_ROOT" && rg -n -e "$legacy_pattern" \
    "${canonical_roots[@]}" -g '*.lean' || true)" &&
    [[ -n "$legacy_hits" ]]; then
  echo "[ownership] legacy flat-domain protocol leaked into canonical ConcretePhi81:" >&2
  echo "$legacy_hits" >&2
  status=1
fi

if (( status != 0 )); then
  exit "$status"
fi

echo "[ownership] active protocol/proof/R1CS ownership contracts passed"
