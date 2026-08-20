import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonReplayAuthority

/-!
Contract: complete authoritative Poseidon2 replay for the four production
PiRLC family runs.

Assurance tier: artifact-checked same-assignment replay semantics for the
Nightstream b2/k16 profile.

Owns: composition of retained call transitions, exact authoritative word
order, and the final even-run tail into one monolithic replay equality.

Does not own: initial or final state placement, final matrix-slice identity,
complete PiRLC algebra, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompleteReplay

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayAuthority
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplaySequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement

/-- State after the independently checked calls and the exact unpermuted
tail. The separate placement bridge must bind this state to final columns. -/
def completedState (run : Run)
    (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F) : Poseidon2Duplex.State :=
  Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
    (finalTailWords run freshWord)
    (callState run (run.raw.callCount - 1) assignment)

/-- One retained input run computes the monolithic replay of the exact
918-word authoritative input frame. -/
theorem input_run_complete_replay
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (inputWordAt assignment)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (inputReplayWords assignment) (initialState run assignment) =
      completedState run assignment (inputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  rw [← input_run_words_exact run selected assignment transition]
  rw [Poseidon2Duplex.absorbSlice_append]
  rw [run_replay_exact run productionRun assignment
    (inputWordAt assignment) transition]
  rfl

/-- One retained output run computes the monolithic replay of the exact
54-word authoritative output frame. -/
theorem output_run_complete_replay
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (transition : RunReplayTransition run assignment (outputWordAt assignment)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (outputReplayWords assignment) (initialState run assignment) =
      completedState run assignment (outputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  rw [← output_run_words_exact run selected assignment transition]
  rw [Poseidon2Duplex.absorbSlice_append]
  rw [run_replay_exact run productionRun assignment
    (outputWordAt assignment) transition]
  rfl

/-- One retained input run computes the complete authoritative replay from
any semantic start with the exact live rate prefix, capacity, and cursor. -/
theorem input_run_complete_replay_from_placed_start
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior)
    (transition : RunReplayTransition run assignment (inputWordAt assignment)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (inputReplayWords assignment) prior =
      completedState run assignment (inputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  rw [← input_run_words_exact run selected assignment transition]
  rw [Poseidon2Duplex.absorbSlice_append]
  rw [run_replay_from_placed_start_exact run productionRun assignment
    (inputWordAt assignment) transition prior placed]
  rfl

/-- One retained output run computes the complete authoritative replay from
any semantic start with the exact live rate prefix, capacity, and cursor. -/
theorem output_run_complete_replay_from_placed_start
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior)
    (transition : RunReplayTransition run assignment (outputWordAt assignment)) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (outputReplayWords assignment) prior =
      completedState run assignment (outputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  rw [← output_run_words_exact run selected assignment transition]
  rw [Poseidon2Duplex.absorbSlice_append]
  rw [run_replay_from_placed_start_exact run productionRun assignment
    (outputWordAt assignment) transition prior placed]
  rfl

/-- The retained input call rows imply the complete authoritative replay. -/
theorem input_rows_imply_complete_replay
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (inputReplayWords assignment) (initialState run assignment) =
      completedState run assignment (inputWordAt assignment) := by
  exact input_run_complete_replay run selected assignment
    (input_rows_imply_run_replay_transition run selected assignment one
      selectorOne satisfied)

/-- The retained output call rows imply the complete authoritative replay. -/
theorem output_rows_imply_complete_replay
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (outputReplayWords assignment) (initialState run assignment) =
      completedState run assignment (outputWordAt assignment) := by
  exact output_run_complete_replay run selected assignment
    (output_rows_imply_run_replay_transition run selected assignment one
      selectorOne satisfied)

/-- Retained input call rows imply the complete authoritative replay from an
exact replay-relevant semantic start. -/
theorem input_rows_imply_complete_replay_from_placed_start
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (inputReplayWords assignment) prior =
      completedState run assignment (inputWordAt assignment) := by
  exact input_run_complete_replay_from_placed_start run selected assignment
    prior placed
      (input_rows_imply_run_replay_transition run selected assignment one
        selectorOne satisfied)

/-- Retained output call rows imply the complete authoritative replay from an
exact replay-relevant semantic start. -/
theorem output_rows_imply_complete_replay_from_placed_start
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (prior : Poseidon2Duplex.State)
    (placed : ReplayStartPlaced run assignment prior)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (outputReplayWords assignment) prior =
      completedState run assignment (outputWordAt assignment) := by
  exact output_run_complete_replay_from_placed_start run selected assignment
    prior placed
      (output_rows_imply_run_replay_transition run selected assignment one
        selectorOne satisfied)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompleteReplay
