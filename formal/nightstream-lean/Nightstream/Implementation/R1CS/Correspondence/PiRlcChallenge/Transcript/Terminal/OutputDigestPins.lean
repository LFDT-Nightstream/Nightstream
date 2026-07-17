import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.OutputDigestSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule

/-!
Verifier-owned constant semantics for the terminal `Pi_CCS` output-digest
handoff into `Pi_RLC`.

Assurance tier: implementation/R1CS correspondence. This file decodes exact
ordinary equations; constants are not trusted merely because the generated
artifact labels them as pins.

Owns: the catch-up squeeze word; all five independently specified label
fields; the digest field-count word; and their derivation from accepted R1CS
equations under canonical-field assumptions.

Does not own: Poseidon2 semantics, inter-call connectivity, digest authority,
the preceding `Pi_CCS` state, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: every value below follows from an accepted equation and
the verifier's constant-one column. The expected label values themselves come
from the independent byte-packing specification.

| Protocol | Phase | Constraint family | Lean guarantee |
|---|---|---|---|
| `Pi_CCS` | catch-up | squeeze constant | exact word `1` is equation-bound |
| `Pi_RLC` | output bind label | constant pins | exact length and four packed label limbs are equation-bound |
| `Pi_RLC` | output bind header | constant pin | exact digest field count `4` is equation-bound |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestPins

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  simp [rowsIncluded]

private theorem acceptedPins
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (pins : List (Nat × Nat))
    (valuesCanonical : ConstantPins.ValuesCanonical pins)
    (piece : Piece)
    (pieceAccepted : piece.Accepted assignment)
    (piecePayload : piece.payload = .ordinary (ConstantPins.rows pins)) :
    forall pin, pin ∈ pins -> assignment pin.1 = pin.2 := by
  rw [Piece.Accepted, piecePayload, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound valuesCanonical
    (rowsIncluded_self (ConstantPins.rows pins)) canonical one pieceAccepted

theorem catchupSqueezePinsCanonical :
    ConstantPins.ValuesCanonical OutputDigestSchedule.catchupSqueezePins := by
  decide

theorem labelPinsCanonical :
    ConstantPins.ValuesCanonical OutputDigestSchedule.labelPins := by
  decide

theorem fieldCountPinsCanonical :
    ConstantPins.ValuesCanonical OutputDigestSchedule.fieldCountPins := by
  decide

/-- The catch-up owner alone binds the squeeze marker. No later `Pi_RLC`
owner is needed to establish this preceding `Pi_CCS` fact. -/
theorem catchupSqueeze_eq_one
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment) :
    assignment 1713693 = 1 := by
  exact acceptedPins canonical one OutputDigestSchedule.catchupSqueezePins
    catchupSqueezePinsCanonical OutputDigestSchedule.catchupPinPiece
    (catchupAccepted OutputDigestSchedule.catchupPinPiece
      OutputDigestSchedule.catchupPinPiece_mem)
    (by rw [OutputDigestSchedule.catchupPinPiece_eq]; rfl)
    (1713693, 1)
    (by simp [OutputDigestSchedule.catchupSqueezePins])

/-- Exact pin facts grouped by their protocol phase. -/
structure Facts (assignment : Nat -> Nat) : Prop where
  catchupSqueeze : forall pin, pin ∈ OutputDigestSchedule.catchupSqueezePins ->
    assignment pin.1 = pin.2
  label : forall pin, pin ∈ OutputDigestSchedule.labelPins ->
    assignment pin.1 = pin.2
  fieldCount : forall pin, pin ∈ OutputDigestSchedule.fieldCountPins ->
    assignment pin.1 = pin.2

theorem facts
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (rlcAccepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    Facts assignment := by
  refine {
    catchupSqueeze := ?_
    label := ?_
    fieldCount := ?_
  }
  · exact acceptedPins canonical one OutputDigestSchedule.catchupSqueezePins
      catchupSqueezePinsCanonical OutputDigestSchedule.catchupPinPiece
      (catchupAccepted OutputDigestSchedule.catchupPinPiece
        OutputDigestSchedule.catchupPinPiece_mem)
      (by rw [OutputDigestSchedule.catchupPinPiece_eq]; rfl)
  · exact acceptedPins canonical one OutputDigestSchedule.labelPins
      labelPinsCanonical OutputDigestSchedule.labelPiece
      (rlcAccepted OutputDigestSchedule.labelPiece
        OutputDigestSchedule.labelPiece_mem)
      (by rw [OutputDigestSchedule.labelPiece_eq]; rfl)
  · exact acceptedPins canonical one OutputDigestSchedule.fieldCountPins
      fieldCountPinsCanonical OutputDigestSchedule.fieldCountPiece
      (rlcAccepted OutputDigestSchedule.fieldCountPiece
        OutputDigestSchedule.fieldCountPiece_mem)
      (by rw [OutputDigestSchedule.fieldCountPiece_eq]; rfl)

/-- The R1CS label pins equal the independently packed mathematical label. -/
theorem labelValues_match_semantics :
    OutputDigestSchedule.labelPins.map Prod.snd =
      OutputDigestSemantics.inputClaimsDigestLabelNats := by
  rw [OutputDigestSemantics.inputClaimsDigestLabelNats_eq]
  decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestPins
