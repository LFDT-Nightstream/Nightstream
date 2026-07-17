import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Indexed artifact owners for terminal-NC rounds one through fourteen.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the exact affine column and row formulas shared by the fourteen
uniform later rounds; their two message permutations; challenge marker and
permutation; preceding message-length pin; owner membership; independent
Poseidon2-call acceptance; and accepted constant facts.

Does not own: round-zero's distinct three-permutation layout; semantic
message decoding; inter-round state connectivity; SumCheck algebra; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: the formulas in this module are checked against every
finite later-round owner address. Artifact acceptance proves only the
selected equations and Poseidon2 calls. A separate execution refinement must
identify their inputs with the independently replayed typed transcript.

| Stage path | Mathematical obligation | Physical owner | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_14.layout` | every later round follows one exact affine row/column schedule | generated owner pieces | `firstMessagePiece_eq`, `secondMessagePiece_eq`, `squeezePiece_eq` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.length` | the preceding algebra tail pins ten serialized base fields | preceding ordinary piece | `Facts.messageLength` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.permute.0` | length, first three fields, and incoming capacity form the first call | first Poseidon piece | `firstMessageCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.permute.1` | the next four fields form the second call | second Poseidon piece | `secondMessageCallAccepted` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.challenge.marker` | the raw squeeze marker is one | ordinary pin piece | `Facts.squeezeMarker` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.challenge.permute` | the final three fields and marker form the challenge call | third Poseidon piece | `squeezeCallAccepted` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

/-- Number of rounds sharing the later-round physical layout. Round `r`
represents semantic NC round `r+1`. -/
abbrev roundCount : Nat := Schedule.laterRoundCount

/-- Ten coefficient columns advance by one fixed-width polynomial. -/
def coefficientBase (round : Fin roundCount) : Nat :=
  1663457 + 10 * round.val

/-- SSA columns advance by three 600-row Poseidon2 calls plus the fixed
ordinary algebra footprint. -/
def columnBase (round : Fin roundCount) : Nat :=
  1667200 + 1830 * round.val

/-- Local Poseidon2 rows advance by the complete 1,832-row later-round
footprint, including the marker and algebra rows. -/
def localRowBase (round : Fin roundCount) : Nat :=
  3642 + 1832 * round.val

/-- Owner-global rows use the same later-round stride. -/
def ownerRowBase (round : Fin roundCount) : Nat :=
  1616705 + 1832 * round.val

def messageLengthColumn (round : Fin roundCount) : Nat :=
  columnBase round + 36

def firstAllocatedColumn (round : Fin roundCount) : Nat :=
  columnBase round + 37

def firstOutputBase (round : Fin roundCount) : Nat :=
  firstAllocatedColumn round + 592

def secondAllocatedColumn (round : Fin roundCount) : Nat :=
  firstAllocatedColumn round + 600

def secondOutputBase (round : Fin roundCount) : Nat :=
  secondAllocatedColumn round + 592

def squeezeMarkerColumn (round : Fin roundCount) : Nat :=
  secondAllocatedColumn round + 600

def squeezeAllocatedColumn (round : Fin roundCount) : Nat :=
  squeezeMarkerColumn round + 1

def squeezeOutputBase (round : Fin roundCount) : Nat :=
  squeezeAllocatedColumn round + 592

/-- First full-rate message permutation for one later round. -/
def firstMessageCall (round : Fin roundCount) : Poseidon2Call.Call :=
  { rowStart := localRowBase round
    rowEnd := localRowBase round + 600
    inputColumns :=
      [messageLengthColumn round,
       coefficientBase round,
       coefficientBase round + 1,
       coefficientBase round + 2,
       columnBase round + 4,
       columnBase round + 5,
       columnBase round + 6,
       columnBase round + 7]
    firstAllocatedColumn := firstAllocatedColumn round }

/-- Second full-rate message permutation for one later round. -/
def secondMessageCall (round : Fin roundCount) : Poseidon2Call.Call :=
  { rowStart := localRowBase round + 600
    rowEnd := localRowBase round + 1200
    inputColumns :=
      [coefficientBase round + 3,
       coefficientBase round + 4,
       coefficientBase round + 5,
       coefficientBase round + 6,
       firstOutputBase round + 4,
       firstOutputBase round + 5,
       firstOutputBase round + 6,
       firstOutputBase round + 7]
    firstAllocatedColumn := secondAllocatedColumn round }

/-- Final two-field challenge permutation for one later round. -/
def squeezeCall (round : Fin roundCount) : Poseidon2Call.Call :=
  { rowStart := localRowBase round + 1201
    rowEnd := localRowBase round + 1801
    inputColumns :=
      [coefficientBase round + 7,
       coefficientBase round + 8,
       coefficientBase round + 9,
       squeezeMarkerColumn round,
       secondOutputBase round + 4,
       secondOutputBase round + 5,
       secondOutputBase round + 6,
       secondOutputBase round + 7]
    firstAllocatedColumn := squeezeAllocatedColumn round }

/-- Every indexed challenge call exposes its eight consecutive output
columns as the cursor-zero state consumed by the next phase or round. -/
theorem squeezeOutputColumn
    (round : Fin roundCount)
    (lane : Fin PiRlcChallenge.TranscriptMachine.width) :
    (squeezeCall round).columnMap (601 + lane.val) =
      squeezeOutputBase round + lane.val := by
  unfold squeezeCall squeezeOutputBase Poseidon2Call.Call.columnMap
  simp only [List.getD]
  have laneLt := lane.isLt
  simp only [PiRlcChallenge.TranscriptMachine.width] at laneLt
  simp only [show ¬601 + lane.val = 0 by omega, ↓reduceIte]
  simp only [show ¬601 + lane.val < 9 by omega, ↓reduceIte]
  omega

def messageLengthPins (round : Fin roundCount) : List (Nat × Nat) :=
  [(messageLengthColumn round, 10)]

def squeezeMarkerPins (round : Fin roundCount) : List (Nat × Nat) :=
  [(squeezeMarkerColumn round, 1)]

/-- The next message length is the final row of the preceding algebra piece.
For semantic round one this is round zero's algebra; afterward it is the
preceding later-round algebra. -/
def messageLengthPieceIndex (round : Fin roundCount) :
    Fin Rows.pieceCount :=
  ⟨10 + 5 * round.val, by
    have roundLt := round.isLt
    simp only [roundCount, Schedule.laterRoundCount, Rows.pieceCount]
      at roundLt ⊢
    omega⟩

def messageLengthPiece (round : Fin roundCount) : Piece :=
  Rows.pieceAt (messageLengthPieceIndex round)

def firstMessagePiece (round : Fin roundCount) : Piece :=
  Schedule.laterMessageCallPiece round ⟨0, by decide⟩

def secondMessagePiece (round : Fin roundCount) : Piece :=
  Schedule.laterMessageCallPiece round ⟨1, by decide⟩

def squeezeMarkerPiece (round : Fin roundCount) : Piece :=
  Schedule.laterSqueezePinPiece round

def squeezePiece (round : Fin roundCount) : Piece :=
  Schedule.laterSqueezeCallPiece round

/-- Every indexed first message piece is exactly the affine call formula. -/
theorem firstMessagePiece_eq :
    ∀ round : Fin roundCount,
      firstMessagePiece round =
        { rowStart := ownerRowBase round
          rowEnd := ownerRowBase round + 600
          payload := .poseidon (firstMessageCall round) } := by
  decide

/-- Every indexed second message piece is exactly the affine call formula. -/
theorem secondMessagePiece_eq :
    ∀ round : Fin roundCount,
      secondMessagePiece round =
        { rowStart := ownerRowBase round + 600
          rowEnd := ownerRowBase round + 1200
          payload := .poseidon (secondMessageCall round) } := by
  decide

/-- Every indexed challenge piece is exactly the affine call formula. -/
theorem squeezePiece_eq :
    ∀ round : Fin roundCount,
      squeezePiece round =
        { rowStart := ownerRowBase round + 1201
          rowEnd := ownerRowBase round + 1801
          payload := .poseidon (squeezeCall round) } := by
  decide

theorem messageLengthPiece_payload :
    ∀ round : Fin roundCount,
      (messageLengthPiece round).payload =
        .ordinary (messageLengthPiece round).rows := by
  decide

theorem squeezeMarkerPiece_payload :
    ∀ round : Fin roundCount,
      (squeezeMarkerPiece round).payload =
        .ordinary (squeezeMarkerPiece round).rows := by
  decide

theorem messageLengthPins_included :
    ∀ round : Fin roundCount,
      rowsIncluded (ConstantPins.rows (messageLengthPins round))
        (messageLengthPiece round).rows = true := by
  decide

theorem squeezeMarkerPins_included :
    ∀ round : Fin roundCount,
      rowsIncluded (ConstantPins.rows (squeezeMarkerPins round))
        (squeezeMarkerPiece round).rows = true := by
  decide

theorem messageLengthPiece_mem (round : Fin roundCount) :
    messageLengthPiece round ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem firstMessagePiece_mem (round : Fin roundCount) :
    firstMessagePiece round ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem secondMessagePiece_mem (round : Fin roundCount) :
    secondMessagePiece round ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem squeezeMarkerPiece_mem (round : Fin roundCount) :
    squeezeMarkerPiece round ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces :=
  Rows.pieceAt_mem _

theorem squeezePiece_mem (round : Fin roundCount) :
    squeezePiece round ∈
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
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (round : Fin roundCount) :
    TranscriptCertificate.CallAccepted
      (firstMessageCall round) assignment := by
  exact acceptedScheduledCall accepted
    (firstMessagePiece round) (firstMessagePiece_mem round)
    (firstMessageCall round)
    (by rw [firstMessagePiece_eq round])

theorem secondMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (round : Fin roundCount) :
    TranscriptCertificate.CallAccepted
      (secondMessageCall round) assignment := by
  exact acceptedScheduledCall accepted
    (secondMessagePiece round) (secondMessagePiece_mem round)
    (secondMessageCall round)
    (by rw [secondMessagePiece_eq round])

theorem squeezeCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (round : Fin roundCount) :
    TranscriptCertificate.CallAccepted
      (squeezeCall round) assignment := by
  exact acceptedScheduledCall accepted
    (squeezePiece round) (squeezePiece_mem round)
    (squeezeCall round)
    (by rw [squeezePiece_eq round])

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

/-- Exact verifier-owned constants for one indexed later round. -/
structure Facts
    (round : Fin roundCount)
    (assignment : Nat → Nat) : Prop where
  messageLength : assignment (messageLengthColumn round) = 10
  squeezeMarker : assignment (squeezeMarkerColumn round) = 1

theorem facts
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (round : Fin roundCount) :
    Facts round assignment := by
  have lengthFacts :=
    acceptedPins canonical one accepted (messageLengthPins round)
      (by simp [messageLengthPins, ConstantPins.ValuesCanonical, goldilocksP])
      (messageLengthPiece round) (messageLengthPiece_payload round)
      (messageLengthPins_included round) (messageLengthPiece_mem round)
  have markerFacts :=
    acceptedPins canonical one accepted (squeezeMarkerPins round)
      (by simp [squeezeMarkerPins, ConstantPins.ValuesCanonical, goldilocksP])
      (squeezeMarkerPiece round) (squeezeMarkerPiece_payload round)
      (squeezeMarkerPins_included round) (squeezeMarkerPiece_mem round)
  exact {
    messageLength :=
      lengthFacts (messageLengthColumn round, 10)
        (by simp [messageLengthPins])
    squeezeMarker :=
      markerFacts (squeezeMarkerColumn round, 1)
        (by simp [squeezeMarkerPins])
  }

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Artifact
