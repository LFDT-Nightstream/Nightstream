import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact artifact owners for terminal-NC semantic round zero.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the distinct three-permutation message layout; message-length and
challenge-marker pins; the final challenge permutation; exact owner
membership; accepted call proofs; and accepted constant facts.

Does not own: the prologue execution; typed coefficient authority; semantic
round execution; the 30 algebra rows; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: artifact acceptance proves the selected constants and
four exact Poseidon2 calls only. Typed coefficients and the cursor-one
prologue state remain independent inputs to the execution refinement.

| Stage path | Mathematical obligation | Physical owner | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.0.message.length` | pin ten serialized base fields | final prologue ordinary piece | `Facts.messageLength` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.0` | retained tag, length, and first coefficient pair | first message call | `firstMessageCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.1` | coefficient fields two through five | second message call | `secondMessageCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.2` | coefficient fields six through nine | third message call | `thirdMessageCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.round.0.challenge.marker` | raw squeeze marker is one | ordinary pin piece | `Facts.squeezeMarker` |
| `nifs.pi_ccs.nc_sumcheck.round.0.challenge.permute` | marker plus retained lanes one through seven | challenge call | `squeezeCallAccepted` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

def coefficientBase : Nat := 1663447
def messageLengthColumn : Nat := 1664806
def firstAllocatedColumn : Nat := 1664807
def firstOutputBase : Nat := firstAllocatedColumn + 592
def secondAllocatedColumn : Nat := 1665407
def secondOutputBase : Nat := secondAllocatedColumn + 592
def thirdAllocatedColumn : Nat := 1666007
def thirdOutputBase : Nat := thirdAllocatedColumn + 592
def squeezeMarkerColumn : Nat := 1666607
def squeezeAllocatedColumn : Nat := 1666608
def squeezeOutputBase : Nat := squeezeAllocatedColumn + 592

def firstMessageCall : Poseidon2Call.Call :=
  { rowStart := 1210
    rowEnd := 1810
    inputColumns :=
      [Prologue.Artifact.roundTagColumn, messageLengthColumn,
       coefficientBase, coefficientBase + 1,
       Prologue.Artifact.secondOutputBase + 4,
       Prologue.Artifact.secondOutputBase + 5,
       Prologue.Artifact.secondOutputBase + 6,
       Prologue.Artifact.secondOutputBase + 7]
    firstAllocatedColumn := firstAllocatedColumn }

def secondMessageCall : Poseidon2Call.Call :=
  { rowStart := 1810
    rowEnd := 2410
    inputColumns :=
      [coefficientBase + 2, coefficientBase + 3,
       coefficientBase + 4, coefficientBase + 5,
       firstOutputBase + 4, firstOutputBase + 5,
       firstOutputBase + 6, firstOutputBase + 7]
    firstAllocatedColumn := secondAllocatedColumn }

def thirdMessageCall : Poseidon2Call.Call :=
  { rowStart := 2410
    rowEnd := 3010
    inputColumns :=
      [coefficientBase + 6, coefficientBase + 7,
       coefficientBase + 8, coefficientBase + 9,
       secondOutputBase + 4, secondOutputBase + 5,
       secondOutputBase + 6, secondOutputBase + 7]
    firstAllocatedColumn := thirdAllocatedColumn }

def squeezeCall : Poseidon2Call.Call :=
  { rowStart := 3011
    rowEnd := 3611
    inputColumns :=
      [squeezeMarkerColumn,
       thirdOutputBase + 1, thirdOutputBase + 2,
       thirdOutputBase + 3, thirdOutputBase + 4,
       thirdOutputBase + 5, thirdOutputBase + 6,
       thirdOutputBase + 7]
    firstAllocatedColumn := squeezeAllocatedColumn }

def messageLengthPins : List (Nat × Nat) :=
  [(messageLengthColumn, 10)]

def squeezeMarkerPins : List (Nat × Nat) :=
  [(squeezeMarkerColumn, 1)]

def messageLengthPiece : Piece :=
  Schedule.prologuePinPiece ⟨2, by decide⟩

def firstMessagePiece : Piece :=
  Schedule.firstMessageCallPiece ⟨0, by decide⟩

def secondMessagePiece : Piece :=
  Schedule.firstMessageCallPiece ⟨1, by decide⟩

def thirdMessagePiece : Piece :=
  Schedule.firstMessageCallPiece ⟨2, by decide⟩

def squeezeMarkerPiece : Piece :=
  Schedule.firstSqueezePinPiece

def squeezePiece : Piece :=
  Schedule.firstSqueezeCallPiece

theorem firstMessagePiece_eq :
    firstMessagePiece =
      { rowStart := 1614273
        rowEnd := 1614873
        payload := .poseidon firstMessageCall } := by
  decide

theorem secondMessagePiece_eq :
    secondMessagePiece =
      { rowStart := 1614873
        rowEnd := 1615473
        payload := .poseidon secondMessageCall } := by
  decide

theorem thirdMessagePiece_eq :
    thirdMessagePiece =
      { rowStart := 1615473
        rowEnd := 1616073
        payload := .poseidon thirdMessageCall } := by
  decide

theorem squeezePiece_eq :
    squeezePiece =
      { rowStart := 1616074
        rowEnd := 1616674
        payload := .poseidon squeezeCall } := by
  decide

/-- Round zero's challenge call exposes the exact cursor-zero state consumed
by uniform semantic round one. -/
theorem squeezeOutputColumn
    (lane : Fin PiRlcChallenge.TranscriptMachine.width) :
    squeezeCall.columnMap (601 + lane.val) =
      squeezeOutputBase + lane.val := by
  unfold squeezeCall squeezeOutputBase Poseidon2Call.Call.columnMap
  simp only [List.getD]
  have laneLt := lane.isLt
  simp only [PiRlcChallenge.TranscriptMachine.width] at laneLt
  simp only [show ¬601 + lane.val = 0 by omega, ↓reduceIte]
  simp only [show ¬601 + lane.val < 9 by omega, ↓reduceIte]
  omega

theorem messageLengthPiece_payload :
    messageLengthPiece.payload = .ordinary messageLengthPiece.rows := by
  decide

theorem squeezeMarkerPiece_payload :
    squeezeMarkerPiece.payload = .ordinary squeezeMarkerPiece.rows := by
  decide

theorem messageLengthPins_included :
    rowsIncluded (ConstantPins.rows messageLengthPins)
      messageLengthPiece.rows = true := by
  decide

theorem squeezeMarkerPins_included :
    rowsIncluded (ConstantPins.rows squeezeMarkerPins)
      squeezeMarkerPiece.rows = true := by
  decide

theorem messageLengthPiece_mem :
    messageLengthPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem firstMessagePiece_mem :
    firstMessagePiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem secondMessagePiece_mem :
    secondMessagePiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem thirdMessagePiece_mem :
    thirdMessagePiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem squeezeMarkerPiece_mem :
    squeezeMarkerPiece ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem squeezePiece_mem :
    squeezePiece ∈
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

theorem firstMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted firstMessageCall assignment :=
  acceptedScheduledCall accepted firstMessagePiece firstMessagePiece_mem
    firstMessageCall (by rw [firstMessagePiece_eq])

theorem secondMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted secondMessageCall assignment :=
  acceptedScheduledCall accepted secondMessagePiece secondMessagePiece_mem
    secondMessageCall (by rw [secondMessagePiece_eq])

theorem thirdMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted thirdMessageCall assignment :=
  acceptedScheduledCall accepted thirdMessagePiece thirdMessagePiece_mem
    thirdMessageCall (by rw [thirdMessagePiece_eq])

theorem squeezeCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted squeezeCall assignment :=
  acceptedScheduledCall accepted squeezePiece squeezePiece_mem squeezeCall
    (by rw [squeezePiece_eq])

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
  messageLength : assignment messageLengthColumn = 10
  squeezeMarker : assignment squeezeMarkerColumn = 1

theorem facts
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    Facts assignment := by
  have lengthFacts :=
    acceptedPins canonical one accepted messageLengthPins
      (by
        simp [messageLengthPins, ConstantPins.ValuesCanonical, goldilocksP])
      messageLengthPiece messageLengthPiece_payload
      messageLengthPins_included messageLengthPiece_mem
  have markerFacts :=
    acceptedPins canonical one accepted squeezeMarkerPins
      (by
        simp [squeezeMarkerPins, ConstantPins.ValuesCanonical, goldilocksP])
      squeezeMarkerPiece squeezeMarkerPiece_payload
      squeezeMarkerPins_included squeezeMarkerPiece_mem
  exact {
    messageLength :=
      lengthFacts (messageLengthColumn, 10)
        (by simp [messageLengthPins])
    squeezeMarker :=
      markerFacts (squeezeMarkerColumn, 1)
        (by simp [squeezeMarkerPins])
  }

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Artifact
