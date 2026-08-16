import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingProductionSetup
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigest
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact coordinate-commitment transition for the generated streaming
claim-replay arms.

Owns the equality between the Rust-emitted compact seed block and the fixed
verifier setup, placement of claim chunk zero in the 21,220-field PiCCS
coordinate vector, the exact 108-field partial commitment, the zero initial
accumulator, its additive update, and terminal accumulator carry.

Does not own the other selected claim chunks, the 86-step schedule, equality
of supplied and authoritative claim frames, Module-SIS hardness, or recursive
lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingProductionSetup
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
open Nightstream.Implementation.R1CS.Program

def outputWidth : Nat := shape.rows * shape.degree

def firstChunk : Fin claimChunkCount := ⟨0, by decide⟩

/-- Canonical field vector supplied by one physical claim chunk. Fields not
selected by that chunk are zero. -/
def chunkFields
    (chunkIndex : Fin claimChunkCount) (chunk : List Nat) : Fields :=
  fun field =>
    if selected : claimChunk field = chunkIndex then
      ⟨chunk.getD (claimChunkOffset field).val 0 % goldilocksP,
        Nat.mod_lt _ (by decide)⟩
    else
      ⟨0, by decide⟩

/-- Exact verifier-owned partial coordinate commitment for one claim chunk. -/
def partialCommitment
    (chunkIndex : Fin claimChunkCount) (chunk : List Nat) : OutputFields :=
  maskedConcreteBinding productionSetup (chunkFields chunkIndex chunk)
    (chunkMask chunkIndex)

/-- Physical values in the generated full arm's 1,024-field chunk. -/
def fullChunkValues (assignment : Nat → Nat) : List Nat :=
  (List.range claimChunkWidth).map fun index =>
    assignment (chunkColumn fullArm index)

/-- Physical state column of one carried coordinate commitment field. -/
def commitmentColumn
    (kind : ArmKind) (side : StateSide) (output : Fin outputWidth) : Nat :=
  stateWordColumnFor kind side (20 + output.val)

/-- The sole generated coordinate call in the chunk-zero full arm. -/
def fullCoordinateCall : CoordinateCall :=
  fullArm.coordinateCalls.getD 0 default

def fullPartialColumn (output : Fin outputWidth) : Nat :=
  fullCoordinateCall.layout.outputColumn output

theorem outputWidth_exact : outputWidth = 108 := by
  decide

theorem fullCoordinateCall_mem :
    fullCoordinateCall ∈ fullArm.coordinateCalls := by
  native_decide

theorem fullCoordinateCall_chunk :
    fullCoordinateCall.chunk = firstChunk := by
  native_decide

theorem fullCoordinateCall_for_chunk :
    ForClaimChunk fullCoordinateCall.layout firstChunk := by
  unfold ForClaimChunk
  native_decide

/-- The compact Rust seed schedule is exactly the verifier-owned production
setup. The generated seed bytes are not independent commitment authority. -/
theorem fullCoordinateCall_block_exact :
    fullCoordinateCall.block =
      coordinateBlock productionSetup fullCoordinateCall.layout := by
  native_decide

theorem fullCoordinateCall_rows_exact :
    fullCoordinateCall.rows =
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
        productionSetup fullCoordinateCall.layout := by
  unfold CoordinateCall.rows
    Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
  rw [fullCoordinateCall_block_exact]

theorem fullCoordinateCall_chunkBase :
    fullCoordinateCall.chunkBase = chunkBase fullArm := by
  native_decide

theorem fullLayout_fieldColumn (field : Fin fieldCount) :
    fullCoordinateCall.layout.fieldColumn field =
      chunkColumn fullArm (claimChunkOffset field).val := by
  simp [CoordinateCall.layout, chunkColumn, fullCoordinateCall_chunkBase]

theorem fullChunkValues_getD
    (assignment : Nat → Nat) (index : Nat)
    (bound : index < claimChunkWidth) :
    (fullChunkValues assignment).getD index 0 =
      assignment (chunkColumn fullArm index) := by
  unfold fullChunkValues
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem (by simpa using bound)]
  simp [List.getElem_range]

theorem full_active_fields_placed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ActiveFieldsPlaced fullCoordinateCall.layout assignment
      (chunkFields firstChunk (fullChunkValues assignment)) := by
  intro field active
  have selected : claimChunk field = firstChunk := by
    exact (mem_activeFields firstChunk field).mp
      (fullCoordinateCall_for_chunk ▸ active)
  rw [fullLayout_fieldColumn]
  simp only [chunkFields, dif_pos selected]
  rw [fullChunkValues_getD assignment (claimChunkOffset field).val
    (claimChunkOffset field).isLt]
  exact (Nat.mod_eq_of_lt (canonical _)).symm

private def glueProgram (kind : ArmKind) : List Row :=
  (armFor kind).glueRows.map IndexedRow.row

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem included_normalized_satisfies
    (kind : ArmKind) (rows : List Row) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment)
    (included : rowsIncluded
      (Poseidon2Normalized.normalizeProgram rows) (glueProgram kind) = true) :
    Satisfies rows assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram rows assignment).mp
  intro row member
  exact glue_satisfies kind assignment satisfied row
    (rowsIncluded_sound included row member)

/-- Satisfaction of the generated coordinate owner gives satisfaction of
the verifier-owned complete coordinate program. -/
theorem full_coordinate_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : fullArm.Satisfied assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
        productionSetup fullCoordinateCall.layout)
      assignment := by
  rw [← fullCoordinateCall_rows_exact]
  exact coordinate_call_holds fullArm assignment satisfied fullCoordinateCall
    fullCoordinateCall_mem

/-- The generated coordinate rows determine the exact chunk-zero partial
commitment from the same physical chunk fields used by Poseidon2 replay. -/
theorem full_partial_commitment
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (fullPartialColumn output) =
        (partialCommitment firstChunk (fullChunkValues assignment) output).val := by
  have exact := rows_imply_claimChunkCommitment firstChunk
    fullCoordinateCall_for_chunk canonical one
    (full_active_fields_placed assignment canonical)
    (full_coordinate_rows_satisfy assignment satisfied)
  intro output
  let pair := outputPair output
  have atPair := exact pair.1 pair.2
  have indexExact : outputIndex pair.1 pair.2 = output := by
    exact Equiv.apply_symm_apply _ output
  simpa [fullPartialColumn, partialCommitment, pair, indexExact] using atPair

def fullBeforePins : List (Nat × Nat) :=
  List.ofFn fun output : Fin outputWidth =>
    (commitmentColumn .full .before output, 0)

private theorem fullBeforePins_canonical :
    ConstantPins.ValuesCanonical fullBeforePins := by
  native_decide

private theorem fullBeforePins_in_glue :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows fullBeforePins))
      (glueProgram .full) = true := by
  native_decide

/-- The generated first selected arm starts the carried coordinate
commitment from the all-zero accumulator. -/
theorem full_before_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (commitmentColumn .full .before output) = 0 := by
  have compactSatisfies := included_normalized_satisfies .full
    (ConstantPins.rows fullBeforePins) assignment satisfied
    fullBeforePins_in_glue
  have facts := ConstantPins.sound fullBeforePins_canonical
    (by native_decide) canonical one compactSatisfies
  intro output
  exact facts (commitmentColumn .full .before output, 0)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)

def fullUpdateRow (output : Fin outputWidth) : Row :=
  ⟨[(commitmentColumn .full .before output, goldilocksP - 1),
    (commitmentColumn .full .after output, 1),
    (fullPartialColumn output, goldilocksP - 1)], [(0, 1)], []⟩

def fullUpdateRows : List Row := List.ofFn fullUpdateRow

private theorem fullUpdateRows_in_glue :
    rowsIncluded fullUpdateRows (glueProgram .full) = true := by
  native_decide

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem fullUpdateRow_perms :
    ∀ output : Fin outputWidth,
      (fullUpdateRow output).a.Perm
          (builderLinearRow (commitmentColumn .full .after output)
            [(commitmentColumn .full .before output, 1),
              (fullPartialColumn output, 1)]).a ∧
        (fullUpdateRow output).b.Perm
          (builderLinearRow (commitmentColumn .full .after output)
            [(commitmentColumn .full .before output, 1),
              (fullPartialColumn output, 1)]).b ∧
        (fullUpdateRow output).c.Perm
          (builderLinearRow (commitmentColumn .full .after output)
            [(commitmentColumn .full .before output, 1),
              (fullPartialColumn output, 1)]).c := by
  native_decide

theorem full_update_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (commitmentColumn .full .after output) =
        (assignment (commitmentColumn .full .before output) +
          assignment (fullPartialColumn output)) % goldilocksP := by
  have compactSatisfies : Satisfies fullUpdateRows assignment := by
    intro row member
    exact glue_satisfies .full assignment satisfied row
      (rowsIncluded_sound fullUpdateRows_in_glue row member)
  intro output
  have emitted := compactSatisfies (fullUpdateRow output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have builderHolds := rowHolds_of_operand_perms assignment
    (fullUpdateRow_perms output).1 (fullUpdateRow_perms output).2.1
    (fullUpdateRow_perms output).2.2 emitted
  have defined := builderLinearRow_sound canonical one
    (commitmentColumn .full .after output)
    [(commitmentColumn .full .before output, 1),
      (fullPartialColumn output, 1)]
    (by simp [CanonicalTerms]; decide) builderHolds
  simpa [lcEval, Nat.add_comm, Nat.mul_comm] using defined

/-- The full arm adds the exact verifier-owned chunk-zero partial commitment
to every carried accumulator coordinate. -/
theorem full_commitment_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (commitmentColumn .full .after output) =
        (assignment (commitmentColumn .full .before output) +
          (partialCommitment firstChunk (fullChunkValues assignment) output).val) %
            goldilocksP := by
  intro output
  rw [full_update_facts assignment canonical one satisfied output]
  rw [full_partial_commitment assignment canonical one satisfied output]

def finalCarryPairs : List (Nat × Nat) :=
  List.ofFn fun output : Fin outputWidth =>
    (commitmentColumn .final .after output,
      commitmentColumn .final .before output)

private theorem finalCarryPairs_in_glue :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram
        (EqualityPins.rows finalCarryPairs))
      (glueProgram .final) = true := by
  native_decide

/-- The terminal arm carries the full 108-field coordinate commitment
without change. -/
theorem final_commitment_carry
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (commitmentColumn .final .after output) =
        assignment (commitmentColumn .final .before output) := by
  have compactSatisfies :
      Satisfies (EqualityPins.rows finalCarryPairs) assignment := by
    exact included_normalized_satisfies .final
      (EqualityPins.rows finalCarryPairs) assignment satisfied
      finalCarryPairs_in_glue
  have facts := EqualityPins.rows_sound canonical one compactSatisfies
  intro output
  exact facts
    (commitmentColumn .final .after output,
      commitmentColumn .final .before output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate
