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
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Projection/IndexedRows.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Projection/ArtifactProgram.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/IndexedRows.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/ArtifactProgram.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeSelectiveFixedPoint.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeSelectiveFixedPoint
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

# Active source/consumer proofs and their reusable Phi81 arithmetic must not
# reach mathematical facts through historical fixed-profile artifacts. The
# legacy wrappers may depend on these neutral modules, never the reverse.
profile_neutral_roots=(
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Projection/IndexedRows.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Projection/ArtifactProgram.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/IndexedRows.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/ArtifactProgram.lean
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/Phi81
  formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/PiCcsOutputDigest/ActiveSourceLayout
)
diagnostic_import_pattern='^import Nightstream\.Implementation\.R1CS\.(Artifacts\.FPrimeRecursive|Ownership\.FPrime(FullHistory|Recursive)|Correspondence\.FPrime(FullHistory|Recursive))'
if diagnostic_hits="$(cd "$REPO_ROOT" && rg -n \
    -e "$diagnostic_import_pattern" "${profile_neutral_roots[@]}" \
    -g '*.lean' || true)" && [[ -n "$diagnostic_hits" ]]; then
  echo "[ownership] fixed-profile dependency leaked into active/neutral projection proofs:" >&2
  echo "$diagnostic_hits" >&2
  status=1
fi

# The active artifact tree owns retained data and executable certificate
# checks. It must not depend upward on semantic correspondence. Conversely,
# correspondence consumes only the stable artifact facade, never generated
# metadata or row shards directly.
active_artifact_root=formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint
active_correspondence_root=formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeSelectiveFixedPoint
if artifact_import_hits="$(cd "$REPO_ROOT" && rg -n \
    '^import Nightstream\.Implementation\.R1CS\.Correspondence' \
    "$active_artifact_root" -g '*.lean' || true)" &&
    [[ -n "$artifact_import_hits" ]]; then
  echo "[ownership] active artifact imports semantic correspondence:" >&2
  echo "$artifact_import_hits" >&2
  status=1
fi
if generated_import_hits="$(cd "$REPO_ROOT" && rg -n \
    '^import Nightstream\.Implementation\.R1CS\.Artifacts\.FPrimeSelectiveFixedPoint\..*\.Generated' \
    "$active_correspondence_root" -g '*.lean' || true)" &&
    [[ -n "$generated_import_hits" ]]; then
  echo "[ownership] active correspondence bypasses its artifact facade:" >&2
  echo "$generated_import_hits" >&2
  status=1
fi

if (( status != 0 )); then
  exit "$status"
fi

echo "[ownership] active protocol/proof/R1CS ownership contracts passed"
