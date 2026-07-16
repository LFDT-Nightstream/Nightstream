import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.LaneRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.SelectionRows

/-!
Exact production ownership tree for the first recursive-profile `Pi_RLC`
scalar sampler.

Owns: the initialization row carrying the zero accepted-prefix count; the
sixteen exact block/lane row pieces; their counter-chain columns; and the exact
54-of-64 tail piece.

Does not own: the semantic proof of first-accepted selection, aggregation of
the sixteen lane refinements, coefficient assembly, another scalar/profile,
Rust trace conformance, row removal, or whole-circuit cost totals.

Emits constraints: no.

Authority boundary: these artifact objects establish only exact production
placement and owner acceptance. Their mathematical meaning comes separately
from `LaneRows`, `SelectionRows`, and the independent transcript/sampler
semantics.

| Protocol | Phase | Constraint family | Exact production owner | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/init | accepted-prefix zero | rows `353096..353101` | the first lane starts from zero |
| `Pi_RLC` | sampler/block `[0..3]` | four lane leaves per block | sixteen 104-row pieces | indexed lane lookup names the exact rows and counter predecessor |
| `Pi_RLC` | sampler/tail | bounded 54-of-64 selection | rows `358881..361480` | exact mapped tail is owned as one 2,599-row phase |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000

def initialCountColumn : Nat := 350649

def tailBitStarts : List Nat :=
  [351854, 352012, 352170, 352328,
   353090, 353248, 353406, 353564,
   354326, 354484, 354642, 354800,
   355562, 355720, 355878, 356036]

def tailFirstAllocated : Nat := 356194

/-- Exact accepted-prefix predecessor for each block/lane. -/
def cumPrev (block lane : Fin 4) : Nat :=
  ChunkOrder.select4 block
    (ChunkOrder.select4 lane 350649 352011 352169 352327)
    (ChunkOrder.select4 lane 352485 353247 353405 353563)
    (ChunkOrder.select4 lane 353721 354483 354641 354799)
    (ChunkOrder.select4 lane 354957 355719 355877 356035)

def lanePieceAt (rowStart bitStart predecessor : Nat) : Piece :=
  { rowStart := rowStart
    rowEnd := rowStart + 104
    payload := .ordinary
      (AlphabetSamplingResidualTemplate.laneRows bitStart predecessor) }

/-- Total indexed view of the sixteen exact 104-row production pieces. -/
def lanePiece (block lane : Fin 4) : Piece :=
  ChunkOrder.select4 block
    (ChunkOrder.select4 lane
      (lanePieceAt 354370 351854 350649)
      (lanePieceAt 354543 352012 352011)
      (lanePieceAt 354716 352170 352169)
      (lanePieceAt 354889 352328 352327))
    (ChunkOrder.select4 lane
      (lanePieceAt 355666 353090 352485)
      (lanePieceAt 355839 353248 353247)
      (lanePieceAt 356012 353406 353405)
      (lanePieceAt 356185 353564 353563))
    (ChunkOrder.select4 lane
      (lanePieceAt 356962 354326 353721)
      (lanePieceAt 357135 354484 354483)
      (lanePieceAt 357308 354642 354641)
      (lanePieceAt 357481 354800 354799))
    (ChunkOrder.select4 lane
      (lanePieceAt 358258 355562 354957)
      (lanePieceAt 358431 355720 355719)
      (lanePieceAt 358604 355878 355877)
      (lanePieceAt 358777 356036 356035))

/-- The exact ordinary owner piece that initializes the accepted-prefix
counter. Only its first equation is interpreted here. -/
def initializationPiece : Piece :=
  { rowStart := 353096
    rowEnd := 353101
    payload := .ordinary
      [ ⟨[(350649, 1)], [(0, 1)], []⟩,
        ⟨[(350650, 1), (0, 18446744069414584319)], [(0, 1)], []⟩,
        ⟨[(350651, 1), (0, 18446744069414584320)], [(0, 1)], []⟩,
        ⟨[(350652, 1)], [(0, 1)], []⟩,
        ⟨[(350653, 1), (0, 18446744069414584320)], [(0, 1)], []⟩ ] }

/-- Exact bounded-selection tail following the sixteen lane leaves. It is
selected by position from the generated owner list so membership does not
re-run equality over all 2,599 rows. Its payload is proved separately below. -/
def tailPiece : Piece :=
  FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0.get
    ⟨47, by decide⟩

private theorem fin4_value_cases (index : Fin 4) :
    index.val = 0 ∨ index.val = 1 ∨
      index.val = 2 ∨ index.val = 3 := by
  have indexLt := index.isLt
  omega

theorem lanePiece_payload (block lane : Fin 4) :
    (lanePiece block lane).payload =
      .ordinary (AlphabetSamplingResidualTemplate.laneRows
        (ChunkOrder.bitStart block lane) (cumPrev block lane)) := by
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [lanePiece, lanePieceAt, ChunkOrder.bitStart, cumPrev,
      ChunkOrder.select4, hb, hl]

theorem lanePiece_mem (block lane : Fin 4) :
    lanePiece block lane ∈
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [lanePiece, lanePieceAt, ChunkOrder.select4, hb, hl] <;>
    decide

theorem initializationPiece_mem : initializationPiece ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  decide

theorem tailPiece_payload : tailPiece.payload =
    .ordinary (AlphabetSamplingResidualTemplate.tailRows
      tailBitStarts tailFirstAllocated) := by
  rfl

theorem tailPiece_mem : tailPiece ∈
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner.pieces := by
  simpa [tailPiece,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.owner,
    FPrimeFullHistoryRecursivePiRlcTranscriptRhos.pieces] using
      List.get_mem
        FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Generated.pieces0
        ⟨47, by decide⟩

/-- Exact owner acceptance exposes precisely the mapped 104 rows for one
indexed lane. No neighboring row or generated count is used. -/
theorem accepted_laneRows
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block lane : Fin 4) :
    Satisfies
      (AlphabetSamplingResidualTemplate.laneRows
        (ChunkOrder.bitStart block lane) (cumPrev block lane))
      assignment := by
  have pieceAccepted := accepted (lanePiece block lane)
    (lanePiece_mem block lane)
  rw [Piece.Accepted, lanePiece_payload, Payload.Accepted] at pieceAccepted
  exact pieceAccepted

/-- Exact owner acceptance exposes the complete mapped 2,599-row tail. This
theorem intentionally assigns no first-accepted semantics to those rows. -/
theorem accepted_tailRows
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    Satisfies
      (AlphabetSamplingResidualTemplate.tailRows
        tailBitStarts tailFirstAllocated) assignment := by
  have pieceAccepted := accepted tailPiece tailPiece_mem
  rw [Piece.Accepted, tailPiece_payload, Payload.Accepted] at pieceAccepted
  exact pieceAccepted

/-- The first initialization equation and canonicality force the verifier's
accepted-prefix counter to start at the integer zero. -/
theorem accepted_initialCount_zero
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    assignment initialCountColumn = 0 := by
  change assignment 350649 = 0
  have pieceAccepted := accepted initializationPiece initializationPiece_mem
  rw [Piece.Accepted, initializationPiece, Payload.Accepted] at pieceAccepted
  have rowHolds := pieceAccepted
    ⟨[(350649, 1)], [(0, 1)], []⟩ (by simp)
  have valueCanonical := canonical 350649
  simpa [EquationHolds, lcEval, one,
    Nat.mod_eq_of_lt valueCanonical] using rowHolds

/-- Every lane after the first consumes the previous lane's exact final-count
column. The final `(3,3)` branch names the counter consumed by the tail. -/
theorem counter_chain (block lane : Fin 4) :
    cumPrev block lane =
      if block.val = 0 ∧ lane.val = 0 then initialCountColumn
      else if lane.val = 0 then
        LaneRows.finalCountColumn
          (ChunkOrder.bitStart
            ⟨block.val - 1, by
              have blockLt := block.isLt
              omega⟩
            ⟨3, by decide⟩)
      else
        LaneRows.finalCountColumn
          (ChunkOrder.bitStart block
            ⟨lane.val - 1, by
              have laneLt := lane.isLt
              omega⟩) := by
  rcases fin4_value_cases block with hb | hb | hb | hb <;>
    rcases fin4_value_cases lane with hl | hl | hl | hl <;>
    simp [cumPrev, initialCountColumn, LaneRows.finalCountColumn,
      ChunkOrder.bitStart, ChunkOrder.select4, hb, hl]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalarRows
