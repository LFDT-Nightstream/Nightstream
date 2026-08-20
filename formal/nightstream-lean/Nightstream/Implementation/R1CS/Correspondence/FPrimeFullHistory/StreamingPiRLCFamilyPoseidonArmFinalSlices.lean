import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyConcretePhaseBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonRunFinalSlice

/-!
Contract: one finite relation for the selected PiRLC Poseidon2 input and
output runs.

Assurance tier: artifact-checked row-family subrelation for the Nightstream
b2/k16 profile.

Owns: the exact final matrix action for every Rust-emitted Poseidon2 row in
the selected arm, and its direct connection to the concrete family phase.

Does not own: the complete production relation, non-Poseidon2 rows, relation
satisfaction, lifecycle semantics, collision resistance, or row removal.

Emits constraints: no new rows. It interprets the retained emitted rows.
-/

set_option autoImplicit false
set_option compiler.extract_closed false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonArmFinalSlices

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafFinalSlice
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunFinalSlice
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyConcretePhaseBridge
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority

abbrev profileFinalRows : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.finalRows

theorem profileFinalRows_exact : profileFinalRows = 491046 :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.finalRows_exact

/-- The output run starts at the first row after the selected input run. -/
theorem selected_run_boundary (arm : Arm) :
    (outputRun arm).raw.emittedRowStart =
      (inputRun arm).raw.emittedRowStart +
        (inputRun arm).raw.callCount * 86 := by
  cases arm <;> rfl

/-- Output rows cannot be decoded as input rows. This is the sole priority
separation fact used by the shared relation. -/
theorem output_block_row_not_input_owned
    (arm : Arm) (index : Fin (outputRun arm).raw.callCount)
    (offset : Fin ((outputRun arm).emittedBlockAt index).rows.length) :
    ¬ RowOwned (inputRun arm)
      (((outputRun arm).emittedBlockAt index).finalRowStart + offset.val) := by
  intro owned
  have outputLower :
      (outputRun arm).raw.emittedRowStart <=
        ((outputRun arm).emittedBlockAt index).finalRowStart + offset.val := by
    change
      (outputRun arm).raw.emittedRowStart <=
        (outputRun arm).raw.emittedRowStart + index.val *
            Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
          offset.val
    omega
  rw [selected_run_boundary arm] at outputLower
  exact (Nat.not_lt_of_ge outputLower) owned.2

theorem input_block_rows_fit
    (arm : Arm) (index : Fin (inputRun arm).raw.callCount) :
    ((inputRun arm).emittedBlockAt index).finalRowStart +
        ((inputRun arm).emittedBlockAt index).rows.length <=
      profileFinalRows := by
  have indexLt := index.isLt
  change
    (inputRun arm).raw.emittedRowStart + index.val *
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
        (rowsFor ((inputRun arm).leafClassAt index.val)).length <=
      profileFinalRows
  rw [rowsFor_length]
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows_exact]
  rw [profileFinalRows_exact]
  cases arm <;>
    norm_num [inputRun, evenInputRun, oddInputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput]
      at indexLt ⊢ <;>
    omega

theorem output_block_rows_fit
    (arm : Arm) (index : Fin (outputRun arm).raw.callCount) :
    ((outputRun arm).emittedBlockAt index).finalRowStart +
        ((outputRun arm).emittedBlockAt index).rows.length <=
      profileFinalRows := by
  have indexLt := index.isLt
  change
    (outputRun arm).raw.emittedRowStart + index.val *
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
        (rowsFor ((outputRun arm).leafClassAt index.val)).length <=
      profileFinalRows
  rw [rowsFor_length]
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows_exact]
  rw [profileFinalRows_exact]
  cases arm <;>
    norm_num [outputRun, evenOutputRun, oddOutputRun,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput]
      at indexLt ⊢ <;>
    omega

/-- One relation contains both selected runs. Rows outside them are zero.
The witness remains proof-local so compilation does not enumerate the fixed
production row or column domains. -/
theorem armRelation_exists (arm : Arm) :
    ∃ relation : InterpretedRelation profileFinalRows productionFinalColumns,
      (∀ assignment : Fin productionFinalColumns -> F,
        ∀ index : Fin (inputRun arm).raw.callCount,
          FinalRowSliceExact ((inputRun arm).emittedBlockAt index)
            relation assignment) ∧
      (∀ assignment : Fin productionFinalColumns -> F,
        ∀ index : Fin (outputRun arm).raw.callCount,
          FinalRowSliceExact ((outputRun arm).emittedBlockAt index)
            relation assignment) := by
  let relation : InterpretedRelation profileFinalRows productionFinalColumns :=
    { matrices := fun role row column =>
        match rowCombinationAt (columns := productionFinalColumns)
            (inputRun arm) role row.val with
        | some combination => combination column
        | none =>
            match rowCombinationAt (columns := productionFinalColumns)
                (outputRun arm) role row.val with
            | some combination => combination column
            | none => 0 }
  have inputExact
      (assignment : Fin productionFinalColumns -> F)
      (index : Fin (inputRun arm).raw.callCount) :
      FinalRowSliceExact ((inputRun arm).emittedBlockAt index)
        relation assignment := by
    let rowsFit := input_block_rows_fit arm index
    have matrixRow
        (offset : Fin ((inputRun arm).emittedBlockAt index).rows.length)
        (port : Fin 13) :
        relation.matrixAt port
            (finalRowIndex ((inputRun arm).emittedBlockAt index)
              rowsFit offset) =
          portCombination ((inputRun arm).emittedBlockAt index).site
            ((((inputRun arm).emittedBlockAt index).rows.get offset).port
              port) := by
      funext column
      change
        (match rowCombinationAt (columns := productionFinalColumns)
            (inputRun arm) (Role.ofIndex port)
              (((inputRun arm).emittedBlockAt index).finalRowStart +
                offset.val) with
        | some combination => combination column
        | none =>
            match rowCombinationAt (columns := productionFinalColumns)
                (outputRun arm) (Role.ofIndex port)
                  (((inputRun arm).emittedBlockAt index).finalRowStart +
                    offset.val) with
            | some combination => combination column
            | none => 0) =
          portCombination ((inputRun arm).emittedBlockAt index).site
            ((((inputRun arm).emittedBlockAt index).rows.get offset).port
              port) column
      rw [rowCombinationAt_emitted_block (columns := productionFinalColumns)
        (inputRun arm) index offset (Role.ofIndex port)]
      rw [Role.index_ofIndex]
    refine
      { rowsFit := rowsFit
        pointExact := ?_ }
    intro offset
    funext port
    unfold rowPoint matrixImageAt
    change
      DirectRows.LinearCombination.eval
          (relation.matrixAt port
            (finalRowIndex ((inputRun arm).emittedBlockAt index)
              rowsFit offset)) assignment =
        absolutePortAction ((inputRun arm).emittedBlockAt index).site assignment
          ((((inputRun arm).emittedBlockAt index).rows.get offset).port port)
    rw [matrixRow, portCombination_eval, portActionAt_production]
  have outputExact
      (assignment : Fin productionFinalColumns -> F)
      (index : Fin (outputRun arm).raw.callCount) :
      FinalRowSliceExact ((outputRun arm).emittedBlockAt index)
        relation assignment := by
    let rowsFit := output_block_rows_fit arm index
    have matrixRow
        (offset : Fin ((outputRun arm).emittedBlockAt index).rows.length)
        (port : Fin 13) :
        relation.matrixAt port
            (finalRowIndex ((outputRun arm).emittedBlockAt index)
              rowsFit offset) =
          portCombination ((outputRun arm).emittedBlockAt index).site
            ((((outputRun arm).emittedBlockAt index).rows.get offset).port
              port) := by
      funext column
      have inputNone :
          rowCombinationAt (columns := productionFinalColumns)
              (inputRun arm) (Role.ofIndex port)
                (((outputRun arm).emittedBlockAt index).finalRowStart +
                  offset.val) = none := by
        unfold rowCombinationAt
        rw [dif_neg (output_block_row_not_input_owned arm index offset)]
      change
        (match rowCombinationAt (columns := productionFinalColumns)
            (inputRun arm) (Role.ofIndex port)
              (((outputRun arm).emittedBlockAt index).finalRowStart +
                offset.val) with
        | some combination => combination column
        | none =>
            match rowCombinationAt (columns := productionFinalColumns)
                (outputRun arm) (Role.ofIndex port)
                  (((outputRun arm).emittedBlockAt index).finalRowStart +
                    offset.val) with
            | some combination => combination column
            | none => 0) =
          portCombination ((outputRun arm).emittedBlockAt index).site
            ((((outputRun arm).emittedBlockAt index).rows.get offset).port
              port) column
      rw [inputNone]
      rw [rowCombinationAt_emitted_block (columns := productionFinalColumns)
        (outputRun arm) index offset (Role.ofIndex port)]
      rw [Role.index_ofIndex]
    refine
      { rowsFit := rowsFit
        pointExact := ?_ }
    intro offset
    funext port
    unfold rowPoint matrixImageAt
    change
      DirectRows.LinearCombination.eval
          (relation.matrixAt port
            (finalRowIndex ((outputRun arm).emittedBlockAt index)
              rowsFit offset)) assignment =
        absolutePortAction ((outputRun arm).emittedBlockAt index).site assignment
          ((((outputRun arm).emittedBlockAt index).rows.get offset).port port)
    rw [matrixRow, portCombination_eval, portActionAt_production]
  exact ⟨relation, inputExact, outputExact⟩

/-- Opaque selected-arm row-family relation. -/
noncomputable def armRelation (arm : Arm) :
    InterpretedRelation profileFinalRows productionFinalColumns :=
  Classical.choose (armRelation_exists arm)

theorem armRelation_input_exact
    (arm : Arm) (assignment : Fin productionFinalColumns -> F)
    (index : Fin (inputRun arm).raw.callCount) :
    FinalRowSliceExact ((inputRun arm).emittedBlockAt index)
      (armRelation arm) assignment :=
  (Classical.choose_spec (armRelation_exists arm)).1 assignment index

theorem armRelation_output_exact
    (arm : Arm) (assignment : Fin productionFinalColumns -> F)
    (index : Fin (outputRun arm).raw.callCount) :
    FinalRowSliceExact ((outputRun arm).emittedBlockAt index)
      (armRelation arm) assignment :=
  (Classical.choose_spec (armRelation_exists arm)).2 assignment index

/-- Satisfaction of the exact selected Poseidon2 row-family relation closes
the matrix-slice premises of the existing concrete phase theorem. -/
theorem arm_poseidon_rows_imply_concrete_phase
    (arm : Arm) (assignment : Fin productionFinalColumns -> F)
    (constantOne : assignment ⟨0, by decide⟩ = 1)
    (algebraAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.ProductionAccepted
        arm assignment)
    (carryAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ProductionAccepted
        arm assignment)
    (residualAccepted :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.ProductionAccepted
        arm assignment)
    (inputSetup : InputBindingSetup)
    (before after : FamilyState) (family : Family)
    (statePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout)
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.carryAssignment
          assignment) before after)
    (strongSet :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.ChallengesInStrongSet
        before.challenges)
    (residualStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.StateColumnsPlaced
        assignment before after)
    (phaseBindingPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized.PhaseBindingPlaced
        inputSetup family
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
          assignment) assignment)
    (cursorExact : before.familyCursor =
      Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.familyOrdinal family)
    (replayStatesPlaced : ReplayStatesPlaced arm assignment before after)
    (rowsSatisfied : AllRowsSatisfied (armRelation arm) assignment) :
    FamilyPhaseRelation inputSetup before after family
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment)
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
        assignment) := by
  exact final_rows_imply_concrete_phase
    (armRelation arm) arm assignment constantOne algebraAccepted carryAccepted
    residualAccepted inputSetup before after family statePlaced strongSet
    residualStatePlaced phaseBindingPlaced cursorExact replayStatesPlaced
    rowsSatisfied (armRelation_input_exact arm assignment)
    (armRelation_output_exact arm assignment)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonArmFinalSlices
