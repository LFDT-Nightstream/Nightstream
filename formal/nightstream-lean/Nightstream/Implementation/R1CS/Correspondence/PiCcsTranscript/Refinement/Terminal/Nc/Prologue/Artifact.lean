import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact artifact owners for the terminal-NC prologue.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the seven fixed fields before the first prologue permutation; the
length-one field before the second permutation; the retained round tag; both
prologue Poseidon2 calls; owner membership; and accepted constant facts.

Does not own: the state entering NC from FE; semantic prologue replay; the
round-zero message; SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: accepted rows establish only these fixed fields and the
two exact permutation calls. A separate execution theorem must bind the
incoming FE state and prove that the independent `ncPrologue` machine follows
this schedule.

| Stage path | Mathematical obligation | Physical owner | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.prologue.constants.0` | encode raw `[8]`, raw `[9]`, zero pair length and values | first ordinary piece | `Facts`, `firstPins_included` |
| `nifs.pi_ccs.nc_sumcheck.prologue.permute.0` | first four prologue words plus retained FE capacity | first Poseidon piece | `firstCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.prologue.constants.1` | encode raw `[10]` length and retained tag | second ordinary piece | `Facts`, `secondPins_included` |
| `nifs.pi_ccs.nc_sumcheck.prologue.permute.1` | zero-pair message and next length word | second Poseidon piece | `secondCallAccepted` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

def zeroC0Column : Nat := 1663597
def zeroC1Column : Nat := 1663598
def domainLengthColumn : Nat := 1663599
def domainTagColumn : Nat := 1663600
def initialTagLengthColumn : Nat := 1663601
def initialTagColumn : Nat := 1663602
def zeroLengthColumn : Nat := 1663603
def roundTagLengthColumn : Nat := 1664204
def roundTagColumn : Nat := 1664205

def afterFeColumnBase : Nat := 1663411
def firstAllocatedColumn : Nat := 1663604
def firstOutputBase : Nat := firstAllocatedColumn + 592
def secondAllocatedColumn : Nat := 1664206
def secondOutputBase : Nat := secondAllocatedColumn + 592

def firstCall : Poseidon2Call.Call :=
  { rowStart := 7
    rowEnd := 607
    inputColumns :=
      [domainLengthColumn, domainTagColumn,
       initialTagLengthColumn, initialTagColumn,
       afterFeColumnBase + 4, afterFeColumnBase + 5,
       afterFeColumnBase + 6, afterFeColumnBase + 7]
    firstAllocatedColumn := firstAllocatedColumn }

def secondCall : Poseidon2Call.Call :=
  { rowStart := 609
    rowEnd := 1209
    inputColumns :=
      [zeroLengthColumn, zeroC0Column, zeroC1Column,
       roundTagLengthColumn,
       firstOutputBase + 4, firstOutputBase + 5,
       firstOutputBase + 6, firstOutputBase + 7]
    firstAllocatedColumn := secondAllocatedColumn }

def firstPins : List (Nat × Nat) :=
  [(zeroC0Column, 0), (zeroC1Column, 0),
   (domainLengthColumn, 1), (domainTagColumn, 8),
   (initialTagLengthColumn, 1), (initialTagColumn, 9),
   (zeroLengthColumn, 2)]

def secondPins : List (Nat × Nat) :=
  [(roundTagLengthColumn, 1), (roundTagColumn, 10)]

def firstPinPiece : Piece :=
  Schedule.prologuePinPiece ⟨0, by decide⟩

def firstCallPiece : Piece :=
  Schedule.prologueCallPiece ⟨0, by decide⟩

def secondPinPiece : Piece :=
  Schedule.prologuePinPiece ⟨1, by decide⟩

def secondCallPiece : Piece :=
  Schedule.prologueCallPiece ⟨1, by decide⟩

theorem firstCallPiece_eq :
    firstCallPiece =
      { rowStart := 1613070
        rowEnd := 1613670
        payload := .poseidon firstCall } := by
  decide

theorem secondCallPiece_eq :
    secondCallPiece =
      { rowStart := 1613672
        rowEnd := 1614272
        payload := .poseidon secondCall } := by
  decide

/-- The accepted second prologue call exposes all eight state lanes at the
contiguous output base consumed by round zero. -/
theorem secondOutputColumn (lane : Fin PiRlcChallenge.TranscriptMachine.width) :
    secondCall.columnMap (601 + lane.val) =
      secondOutputBase + lane.val := by
  unfold secondCall secondOutputBase Poseidon2Call.Call.columnMap
  simp only [List.getD]
  have laneLt := lane.isLt
  simp only [PiRlcChallenge.TranscriptMachine.width] at laneLt
  simp only [show ¬601 + lane.val = 0 by omega, ↓reduceIte]
  simp only [show ¬601 + lane.val < 9 by omega, ↓reduceIte]
  omega

theorem firstPinPiece_payload :
    firstPinPiece.payload = .ordinary firstPinPiece.rows := by
  decide

theorem secondPinPiece_payload :
    secondPinPiece.payload = .ordinary secondPinPiece.rows := by
  decide

theorem firstPins_included :
    rowsIncluded (ConstantPins.rows firstPins) firstPinPiece.rows = true := by
  decide

theorem secondPins_included :
    rowsIncluded (ConstantPins.rows secondPins) secondPinPiece.rows = true := by
  decide

theorem firstPinPiece_mem :
    firstPinPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem firstCallPiece_mem :
    firstCallPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem secondPinPiece_mem :
    secondPinPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem secondCallPiece_mem :
    secondCallPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

private theorem acceptedScheduledCall
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (piece : Piece)
    (member : piece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces)
    (call : Poseidon2Call.Call)
    (payload : piece.payload = .poseidon call) :
    TranscriptCertificate.CallAccepted call assignment := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, payload] at pieceAccepted
  exact pieceAccepted

theorem firstCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted firstCall assignment :=
  acceptedScheduledCall accepted firstCallPiece firstCallPiece_mem firstCall
    (by rw [firstCallPiece_eq])

theorem secondCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted secondCall assignment :=
  acceptedScheduledCall accepted secondCallPiece secondCallPiece_mem secondCall
    (by rw [secondCallPiece_eq])

private theorem acceptedPins
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (pins : List (Nat × Nat))
    (valuesCanonical : ConstantPins.ValuesCanonical pins)
    (piece : Piece)
    (piecePayload : piece.payload = .ordinary piece.rows)
    (included : rowsIncluded (ConstantPins.rows pins) piece.rows = true)
    (member : piece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces) :
    ∀ pin, pin ∈ pins → assignment pin.1 = pin.2 := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, piecePayload, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound valuesCanonical included canonical one pieceAccepted

structure Facts (assignment : Nat → Nat) : Prop where
  zeroC0 : assignment zeroC0Column = 0
  zeroC1 : assignment zeroC1Column = 0
  domainLength : assignment domainLengthColumn = 1
  domainTag : assignment domainTagColumn = 8
  initialTagLength : assignment initialTagLengthColumn = 1
  initialTag : assignment initialTagColumn = 9
  zeroLength : assignment zeroLengthColumn = 2
  roundTagLength : assignment roundTagLengthColumn = 1
  roundTag : assignment roundTagColumn = 10

theorem facts
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    Facts assignment := by
  have firstFacts :=
    acceptedPins canonical one accepted firstPins
      (by simp [firstPins, ConstantPins.ValuesCanonical, goldilocksP])
      firstPinPiece firstPinPiece_payload firstPins_included firstPinPiece_mem
  have secondFacts :=
    acceptedPins canonical one accepted secondPins
      (by simp [secondPins, ConstantPins.ValuesCanonical, goldilocksP])
      secondPinPiece secondPinPiece_payload secondPins_included
      secondPinPiece_mem
  exact {
    zeroC0 := firstFacts (zeroC0Column, 0) (by simp [firstPins])
    zeroC1 := firstFacts (zeroC1Column, 0) (by simp [firstPins])
    domainLength :=
      firstFacts (domainLengthColumn, 1) (by simp [firstPins])
    domainTag := firstFacts (domainTagColumn, 8) (by simp [firstPins])
    initialTagLength :=
      firstFacts (initialTagLengthColumn, 1) (by simp [firstPins])
    initialTag := firstFacts (initialTagColumn, 9) (by simp [firstPins])
    zeroLength := firstFacts (zeroLengthColumn, 2) (by simp [firstPins])
    roundTagLength :=
      secondFacts (roundTagLengthColumn, 1) (by simp [secondPins])
    roundTag := secondFacts (roundTagColumn, 10) (by simp [secondPins])
  }

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Artifact
