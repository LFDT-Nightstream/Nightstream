import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.Refinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding

/-!
Conditional terminal handoff from the recomputed `Pi_CCS` output digest into
the `Pi_RLC` transcript.

Assurance tier: implementation/R1CS composition. This file joins two already
separate refinement results; it does not manufacture authority for either
side of the boundary.

Owns: equality between the four final output-hash lanes and the four fields
absorbed by the terminal `Pi_RLC` transcript; composition through the two
digest-binding boundaries into the initial rho-sampler state.

Does not own: semantic authority of the dynamic `Pi_CCS` output columns;
public-seed-to-SIS-map conformance; Rust/ChaCha or native Poseidon2 parity;
authority of the pre-handoff catch-up state; collision resistance; row
necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `accepted_conditionalDigestHandoff` is intentionally
conditional on three independently accepted owners. Its digest expression is
derived from the typed serialization and exact SIS/Poseidon equations, never
from a prover-supplied digest. The theorem is not a full `Pi_CCS` soundness
claim until the named upstream authority and native-conformance bridges close.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | output lanes | `accepted_digestFieldValues` | the four transcript fields equal the typed serialization after both SIS maps and the exact sponge |
| `Pi_RLC` | digest handoff | two Poseidon2 boundaries | `accepted_conditionalDigestHandoff` | those same four fields reach the audited rho-sampler entry state |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

abbrev CanonicalAssignment (assignment : Nat → Nat) :=
  ∀ column, assignment column < goldilocksP

/-- Pure value computed by the production-shaped typed serialization, exact
two-stage SIS maps, and exact terminal output-digest sponge. -/
def recomputedDigestValue (assignment : Nat → Nat)
    (canonical : CanonicalAssignment assignment) (lane : Nat) : Nat :=
  Poseidon2Sponge.runValueRounds Poseidon.Schedule.trace.rounds
    (Poseidon.EnvelopeSemantics.envelope
      (Sis.Semantics.apply
        (Sis.Refinement.mapOfBlock Sis.ProductionBinding.compressionBlock)
        (Sis.Semantics.apply
          (Sis.Refinement.mapOfBlock Sis.ProductionBinding.primaryBlock)
          (Sis.ProductionBinding.serializedValues assignment canonical))))
    (fun _ => 0) lane

theorem digestColumns_eq_handoff :
    Poseidon.Schedule.digestColumns =
      [2553433, 2553434, 2553435, 2553436] := by
  decide

/-- The four fields named by the `Pi_RLC` handoff are not carried authority:
accepted output-hash equations recompute each one from the typed serialization
and the two exact SIS maps. -/
theorem accepted_digestFieldValues
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (outputAccepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment) :
    ∀ lane : Fin 4,
      (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
        assignment canonical lane).val =
        recomputedDigestValue assignment canonical lane.val := by
  intro lane
  have digest := Poseidon.Refinement.accepted_composedDigest
    prime canonical one outputAccepted lane.val lane.isLt
  rw [digestColumns_eq_handoff] at digest
  simpa [PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest,
    PiRlcChallenge.Transcript.DigestRounds.fieldAt,
    recomputedDigestValue] using digest

/-- Conditional end-to-end composition of the recomputed terminal-output
digest with the exact `Pi_RLC` transcript handoff. The theorem deliberately
retains the output-hash, catch-up, and transcript-owner premises separately so
that no accepted owner silently stands in for paper-level authority. -/
theorem accepted_conditionalDigestHandoff
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (outputAccepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (rlcAccepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    (∀ lane : Fin 4,
      (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
        assignment canonical lane).val =
        recomputedDigestValue assignment canonical lane.val) ∧
    permute
        (PiRlcChallenge.Transcript.DigestRounds.callInputState assignment canonical
          PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.catchupCall
          ⟨1, by decide⟩) =
      PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.postCatchupState
        assignment canonical ∧
    PiRlcChallenge.Transcript.OutputDigestSemantics.appendInputClaimsDigest
        (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.postCatchupState
          assignment canonical)
        (PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.outputDigest
          assignment canonical) =
      PiRlcChallenge.Transcript.Terminal.ScheduleRefinement.initialState
        assignment canonical := by
  have digestFields := accepted_digestFieldValues
    prime canonical one outputAccepted
  have handoff :=
    PiRlcChallenge.Transcript.Terminal.OutputDigestBinding.accepted_refines_outputDigestBinding
      canonical one catchupAccepted rlcAccepted
  exact ⟨digestFields, handoff.1, handoff.2⟩

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff
