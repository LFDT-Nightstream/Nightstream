import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Schedule
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact constant-pin schedule for the recursive-profile Π_RLC scalar sampler.

Owns: the five ordinary artifact pieces surrounding the scheduled Poseidon2
calls, their protocol/phase grouping, exact generated-owner membership, and
the theorem that owner acceptance forces every listed constant value.

Does not own: Poseidon2 call semantics, inter-call wire connectivity,
canonical-u64/chunk meaning, rejection selection, the initial transcript
state, native Rust conformance, or row/cost authority.

Emits constraints: no. This file names and decodes existing artifact rows.

Authority boundary: pin values are derived from independently accepted R1CS
equations plus canonical assignment values and the verifier-owned constant-one
column. They are not trusted because they appear in a generated list.

| Protocol | Phase | Constraint family | Pins forced by the exact piece |
|---|---|---|---|
| `Pi_RLC` | scalar domain | raw-pair words | length `2`, domain `0`, coordinate `0` |
| `Pi_RLC` | block 0 prelude | sampler counter | accepted-coordinate count starts at `0` |
| `Pi_RLC` | block 0 prelude | digest words | length `2`, domain `1`, counter `0`, squeeze word `1` |
| `Pi_RLC` | blocks 1-3 | digest words | `[2, 1, counter, 1]` for counters `1`, `2`, and `3` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.PinSchedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

namespace Artifact

/-- Constants emitted immediately before the scalar-domain boundary call.
The coordinate wire is written after that call, so it is also the next
state's lane-zero wire. -/
def enterScalarPins : List (Nat × Nat) :=
  [(350046, 2), (350047, 0), (350048, 0)]

/-- Constants emitted between the scalar-domain call and block-zero calls.
Column `350649` belongs to sampler selection; the remaining four pins own the
transcript words for digest block zero. -/
def block0PreludePins : List (Nat × Nat) :=
  [(350649, 0), (350650, 2), (350651, 1), (350652, 0), (350653, 1)]

/-- Transcript words for digest block one. -/
def block1Pins : List (Nat × Nat) :=
  [(352486, 2), (352487, 1), (352488, 1), (352489, 1)]

/-- Transcript words for digest block two. -/
def block2Pins : List (Nat × Nat) :=
  [(353722, 2), (353723, 1), (353724, 2), (353725, 1)]

/-- Transcript words for digest block three. -/
def block3Pins : List (Nat × Nat) :=
  [(354958, 2), (354959, 1), (354960, 3), (354961, 1)]

def enterScalarPinPiece : Piece :=
  { rowStart := 352493
    rowEnd := 352496
    payload := .ordinary (ConstantPins.rows enterScalarPins) }

def block0PreludePinPiece : Piece :=
  { rowStart := 353096
    rowEnd := 353101
    payload := .ordinary (ConstantPins.rows block0PreludePins) }

def block1PinPiece : Piece :=
  { rowStart := 354993
    rowEnd := 354997
    payload := .ordinary (ConstantPins.rows block1Pins) }

def block2PinPiece : Piece :=
  { rowStart := 356289
    rowEnd := 356293
    payload := .ordinary (ConstantPins.rows block2Pins) }

def block3PinPiece : Piece :=
  { rowStart := 357585
    rowEnd := 357589
    payload := .ordinary (ConstantPins.rows block3Pins) }

theorem enterScalarPinPiece_mem :
    enterScalarPinPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [enterScalarPinPiece, enterScalarPins, ConstantPins.rows,
    ConstantPins.pinRow, Program.builderLinearRow, Program.negateTerms,
    Program.negCoeff,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0,
    goldilocksP]

theorem block0PreludePinPiece_mem :
    block0PreludePinPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block0PreludePinPiece, block0PreludePins, ConstantPins.rows,
    ConstantPins.pinRow, Program.builderLinearRow, Program.negateTerms,
    Program.negCoeff,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0,
    goldilocksP]

theorem block1PinPiece_mem :
    block1PinPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block1PinPiece, block1Pins, ConstantPins.rows,
    ConstantPins.pinRow, Program.builderLinearRow, Program.negateTerms,
    Program.negCoeff,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0,
    goldilocksP]

theorem block2PinPiece_mem :
    block2PinPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block2PinPiece, block2Pins, ConstantPins.rows,
    ConstantPins.pinRow, Program.builderLinearRow, Program.negateTerms,
    Program.negCoeff,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0,
    goldilocksP]

theorem block3PinPiece_mem :
    block3PinPiece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simp [block3PinPiece, block3Pins, ConstantPins.rows,
    ConstantPins.pinRow, Program.builderLinearRow, Program.negateTerms,
    Program.negCoeff,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0,
    goldilocksP]

end Artifact

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  simp [rowsIncluded]

private theorem acceptedPins
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (pins : List (Nat × Nat))
    (valuesCanonical : ConstantPins.ValuesCanonical pins)
    (piece : Piece)
    (piecePayload : piece.payload = .ordinary (ConstantPins.rows pins))
    (pieceMember : piece ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces) :
    ∀ pin ∈ pins, assignment pin.1 = pin.2 := by
  have pieceAccepted := accepted piece pieceMember
  rw [Piece.Accepted, piecePayload, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound valuesCanonical
    (rowsIncluded_self (ConstantPins.rows pins)) canonical one pieceAccepted

theorem enterScalarPinsCanonical :
    ConstantPins.ValuesCanonical Artifact.enterScalarPins := by
  decide

theorem block0PreludePinsCanonical :
    ConstantPins.ValuesCanonical Artifact.block0PreludePins := by
  decide

theorem block1PinsCanonical :
    ConstantPins.ValuesCanonical Artifact.block1Pins := by
  decide

theorem block2PinsCanonical :
    ConstantPins.ValuesCanonical Artifact.block2Pins := by
  decide

theorem block3PinsCanonical :
    ConstantPins.ValuesCanonical Artifact.block3Pins := by
  decide

/-- Exact constant facts, grouped by the protocol phase that owns them. -/
structure Facts (assignment : Nat → Nat) : Prop where
  enterScalar : ∀ pin ∈ Artifact.enterScalarPins,
    assignment pin.1 = pin.2
  block0Prelude : ∀ pin ∈ Artifact.block0PreludePins,
    assignment pin.1 = pin.2
  block1 : ∀ pin ∈ Artifact.block1Pins,
    assignment pin.1 = pin.2
  block2 : ∀ pin ∈ Artifact.block2Pins,
    assignment pin.1 = pin.2
  block3 : ∀ pin ∈ Artifact.block3Pins,
    assignment pin.1 = pin.2

/-- Owner acceptance forces all transcript/sampler prelude constants. -/
theorem facts
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    Facts assignment := by
  refine {
    enterScalar := ?_
    block0Prelude := ?_
    block1 := ?_
    block2 := ?_
    block3 := ?_
  }
  · exact acceptedPins canonical one accepted Artifact.enterScalarPins
      enterScalarPinsCanonical Artifact.enterScalarPinPiece rfl
      Artifact.enterScalarPinPiece_mem
  · exact acceptedPins canonical one accepted Artifact.block0PreludePins
      block0PreludePinsCanonical Artifact.block0PreludePinPiece rfl
      Artifact.block0PreludePinPiece_mem
  · exact acceptedPins canonical one accepted Artifact.block1Pins
      block1PinsCanonical Artifact.block1PinPiece rfl Artifact.block1PinPiece_mem
  · exact acceptedPins canonical one accepted Artifact.block2Pins
      block2PinsCanonical Artifact.block2PinPiece rfl Artifact.block2PinPiece_mem
  · exact acceptedPins canonical one accepted Artifact.block3Pins
      block3PinsCanonical Artifact.block3PinPiece rfl Artifact.block3PinPiece_mem

variable {assignment : Nat → Nat}

theorem Facts.enterLength (self : Facts assignment) :
    assignment 350046 = 2 :=
  self.enterScalar (350046, 2) (by simp [Artifact.enterScalarPins])

theorem Facts.enterDomain (self : Facts assignment) :
    assignment 350047 = 0 :=
  self.enterScalar (350047, 0) (by simp [Artifact.enterScalarPins])

theorem Facts.enterCoordinate (self : Facts assignment) :
    assignment 350048 = 0 :=
  self.enterScalar (350048, 0) (by simp [Artifact.enterScalarPins])

theorem Facts.block0SelectionCount (self : Facts assignment) :
    assignment 350649 = 0 :=
  self.block0Prelude (350649, 0) (by simp [Artifact.block0PreludePins])

theorem Facts.block0Length (self : Facts assignment) :
    assignment 350650 = 2 :=
  self.block0Prelude (350650, 2) (by simp [Artifact.block0PreludePins])

theorem Facts.block0Domain (self : Facts assignment) :
    assignment 350651 = 1 :=
  self.block0Prelude (350651, 1) (by simp [Artifact.block0PreludePins])

theorem Facts.block0Counter (self : Facts assignment) :
    assignment 350652 = 0 :=
  self.block0Prelude (350652, 0) (by simp [Artifact.block0PreludePins])

theorem Facts.block0Squeeze (self : Facts assignment) :
    assignment 350653 = 1 :=
  self.block0Prelude (350653, 1) (by simp [Artifact.block0PreludePins])

theorem Facts.block1Length (self : Facts assignment) :
    assignment 352486 = 2 :=
  self.block1 (352486, 2) (by simp [Artifact.block1Pins])

theorem Facts.block1Domain (self : Facts assignment) :
    assignment 352487 = 1 :=
  self.block1 (352487, 1) (by simp [Artifact.block1Pins])

theorem Facts.block1Counter (self : Facts assignment) :
    assignment 352488 = 1 :=
  self.block1 (352488, 1) (by simp [Artifact.block1Pins])

theorem Facts.block1Squeeze (self : Facts assignment) :
    assignment 352489 = 1 :=
  self.block1 (352489, 1) (by simp [Artifact.block1Pins])

theorem Facts.block2Length (self : Facts assignment) :
    assignment 353722 = 2 :=
  self.block2 (353722, 2) (by simp [Artifact.block2Pins])

theorem Facts.block2Domain (self : Facts assignment) :
    assignment 353723 = 1 :=
  self.block2 (353723, 1) (by simp [Artifact.block2Pins])

theorem Facts.block2Counter (self : Facts assignment) :
    assignment 353724 = 2 :=
  self.block2 (353724, 2) (by simp [Artifact.block2Pins])

theorem Facts.block2Squeeze (self : Facts assignment) :
    assignment 353725 = 1 :=
  self.block2 (353725, 1) (by simp [Artifact.block2Pins])

theorem Facts.block3Length (self : Facts assignment) :
    assignment 354958 = 2 :=
  self.block3 (354958, 2) (by simp [Artifact.block3Pins])

theorem Facts.block3Domain (self : Facts assignment) :
    assignment 354959 = 1 :=
  self.block3 (354959, 1) (by simp [Artifact.block3Pins])

theorem Facts.block3Counter (self : Facts assignment) :
    assignment 354960 = 3 :=
  self.block3 (354960, 3) (by simp [Artifact.block3Pins])

theorem Facts.block3Squeeze (self : Facts assignment) :
    assignment 354961 = 1 :=
  self.block3 (354961, 1) (by simp [Artifact.block3Pins])

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.PinSchedule
