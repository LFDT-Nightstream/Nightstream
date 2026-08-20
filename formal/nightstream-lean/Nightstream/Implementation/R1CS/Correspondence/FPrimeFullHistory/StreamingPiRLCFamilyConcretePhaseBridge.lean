import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonFinalRowBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonReplayBridge

/-!
Contract: compose the normalized production PiRLC row families and the
selected Poseidon2 final-row slices into one concrete family phase.

Assurance tier: same-assignment semantic composition for the Nightstream
b2/k16 profile.

Owns: the implication from normalized algebra, carry, and residual acceptance,
complete final-relation satisfaction, exact selected Poseidon2 row slices,
and explicit state placement to `FamilyPhaseRelation`.

Does not own: proofs of the matrix-slice premises, relation generation,
lifecycle orchestration, transcript collision resistance, or PiDEC.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyConcretePhaseBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayBridge
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority

/-- Accepted normalized rows and exact final row ownership imply one complete
concrete PiRLC family phase on the same assignment. -/
theorem final_rows_imply_concrete_phase
    {rows : Nat}
    (relation : InterpretedRelation rows productionFinalColumns)
    (arm : Arm) (assignment : Fin productionFinalColumns → F)
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
    (finalSatisfied : AllRowsSatisfied relation assignment)
    (inputSlicesExact : ∀ index : Fin (inputRun arm).raw.callCount,
      FinalRowSliceExact ((inputRun arm).emittedBlockAt index)
        relation assignment)
    (outputSlicesExact : ∀ index : Fin (outputRun arm).raw.callCount,
      FinalRowSliceExact ((outputRun arm).emittedBlockAt index)
        relation assignment) :
    FamilyPhaseRelation inputSetup before after family
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment)
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
        assignment) := by
  have one : absoluteValue assignment 0 = 1 := by
    rw [absoluteValue_of_lt assignment 0 (by decide)]
    exact constantOne
  have selectorOne :
      absoluteValue assignment (inputRun arm).raw.selectorColumn = 1 := by
    rw [inputRun_selectorColumn_eq arm]
    rw [absoluteValue_of_lt assignment _
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.selectorColumn
        arm).isLt]
    exact algebraAccepted.1
  have inputSatisfied : ∀ index : Fin (inputRun arm).raw.callCount,
      ((inputRun arm).emittedBlockAt index).Satisfied assignment := by
    intro index
    exact final_rows_imply_emitted_block_satisfied
      (inputSlicesExact index) finalSatisfied
  have outputSatisfied : ∀ index : Fin (outputRun arm).raw.callCount,
      ((outputRun arm).emittedBlockAt index).Satisfied assignment := by
    intro index
    exact final_rows_imply_emitted_block_satisfied
      (outputSlicesExact index) finalSatisfied
  have replay := rows_imply_replayTransition arm assignment before after
    replayStatesPlaced one selectorOne inputSatisfied outputSatisfied
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.jointAccepted_implies_concrete_phase
      arm assignment constantOne algebraAccepted carryAccepted residualAccepted
      inputSetup before after family statePlaced strongSet residualStatePlaced
      phaseBindingPlaced cursorExact replay

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyConcretePhaseBridge
