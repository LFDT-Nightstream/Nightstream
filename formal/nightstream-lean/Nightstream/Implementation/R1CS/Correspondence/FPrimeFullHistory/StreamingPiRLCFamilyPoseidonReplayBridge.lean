import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCompleteReplay

/-!
Contract: bridge the retained production PiRLC Poseidon2 call rows to the
normalized semantic replay transition.

Assurance tier: artifact-checked same-assignment replay semantics for the
Nightstream b2/k16 profile.

Owns: exact arm-to-run selection and exact carried-state placement at the
semantic replay boundary.

Does not own: algebra, carry, residual, opening, lifecycle, or collision
soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompleteReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplaySequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority

abbrev Arm :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.Arm

def inputRun : Arm → Run
  | .even => evenInputRun
  | .odd => oddInputRun

def outputRun : Arm → Run
  | .even => evenOutputRun
  | .odd => oddOutputRun

theorem inputRun_selected (arm : Arm) :
    inputRun arm = evenInputRun ∨ inputRun arm = oddInputRun := by
  cases arm <;> simp [inputRun]

theorem outputRun_selected (arm : Arm) :
    outputRun arm = evenOutputRun ∨ outputRun arm = oddOutputRun := by
  cases arm <;> simp [outputRun]

theorem run_selector_eq (arm : Arm) :
    (outputRun arm).raw.selectorColumn =
      (inputRun arm).raw.selectorColumn := by
  cases arm <;> rfl

theorem inputRun_selectorColumn_eq (arm : Arm) :
    (inputRun arm).raw.selectorColumn =
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.selectorColumn
        arm).val := by
  cases arm <;> rfl

/-- Exact replay-relevant placement of the two carried semantic states. The
start omits rate lanes that the first absorption overwrites. The end binds
the complete independently checked replay result, not a digest of it. -/
structure ReplayStatesPlaced (arm : Arm)
    (assignment : Fin productionFinalColumns → F)
    (before after : FamilyState) : Prop where
  inputStart : ReplayStartPlaced (inputRun arm) assignment before.inputReplay
  inputAfter : after.inputReplay =
    completedState (inputRun arm) assignment (inputWordAt assignment)
  outputStart : ReplayStartPlaced (outputRun arm) assignment before.outputReplay
  outputAfter : after.outputReplay =
    completedState (outputRun arm) assignment (outputWordAt assignment)

/-- The retained selected input and output call rows imply the complete
normalized semantic replay transition on the same final assignment. -/
theorem rows_imply_replayTransition
    (arm : Arm)
    (assignment : Fin productionFinalColumns → F)
    (before after : FamilyState)
    (placed : ReplayStatesPlaced arm assignment before after)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment (inputRun arm).raw.selectorColumn = 1)
    (inputSatisfied : ∀ index : Fin (inputRun arm).raw.callCount,
      ((inputRun arm).emittedBlockAt index).Satisfied assignment)
    (outputSatisfied : ∀ index : Fin (outputRun arm).raw.callCount,
      ((outputRun arm).emittedBlockAt index).Satisfied assignment) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.ReplayTransition
      before after
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
        assignment)
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
        assignment) := by
  have outputSelectorOne :
      absoluteValue assignment (outputRun arm).raw.selectorColumn = 1 := by
    rw [run_selector_eq arm]
    exact selectorOne
  constructor
  · calc
      after.inputReplay =
          completedState (inputRun arm) assignment
            (inputWordAt assignment) := placed.inputAfter
      _ = Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
          (inputReplayWords assignment) before.inputReplay :=
        (input_rows_imply_complete_replay_from_placed_start
          (inputRun arm) (inputRun_selected arm) assignment
          before.inputReplay placed.inputStart one selectorOne
          inputSatisfied).symm
      _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraInputs
              assignment)) before.inputReplay := by
        rw [inputReplayWords_eq_phaseFields]
        rfl
  · calc
      after.outputReplay =
          completedState (outputRun arm) assignment
            (outputWordAt assignment) := placed.outputAfter
      _ = Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
          (outputReplayWords assignment) before.outputReplay :=
        (output_rows_imply_complete_replay_from_placed_start
          (outputRun arm) (outputRun_selected arm) assignment
          before.outputReplay placed.outputStart one outputSelectorOne
          outputSatisfied).symm
      _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (ringFields
            (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized.algebraOutput
              assignment)) before.outputReplay := by
        rw [outputReplayWords_eq_ringFields]
        rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayBridge
