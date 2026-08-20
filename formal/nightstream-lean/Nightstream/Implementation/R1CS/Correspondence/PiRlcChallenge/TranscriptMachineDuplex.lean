import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.ExtractedReference

/-!
Contract: value-preserving conversion from the compact transcript machine to
the independent Poseidon2 duplex model.

Owns the lane and cursor conversion, one overwrite-absorb step, one
permutation step, and bulk external-column slice execution. The conversion
uses the selected production constants and preserves Rust's eager slice-end
normalization.

Does not own generated rows, a protocol operation schedule, claim-frame
authority, collision resistance, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.PiRlcChallenge
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Forget the finite wrappers after their canonicality and cursor bounds
have been established by the transcript-machine types. -/
def toDuplex (state : TranscriptMachine.State) : Poseidon2Duplex.State where
  lanes := fun lane => (state.lanes lane).val
  absorbed := state.absorbed.val

@[simp] theorem toDuplex_lane
    (state : TranscriptMachine.State) (lane : Fin TranscriptMachine.width) :
    (toDuplex state).lanes lane = (state.lanes lane).val := by
  rfl

@[simp] theorem toDuplex_absorbed (state : TranscriptMachine.State) :
    (toDuplex state).absorbed = state.absorbed.val := by
  rfl

private theorem duplexStateExt {left right : Poseidon2Duplex.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

/-- Both models call the same independently selected Poseidon2 reference
permutation and reset the cursor to zero. -/
theorem permute_toDuplex (state : TranscriptMachine.State) :
    toDuplex (TranscriptMachine.permute state) =
      Poseidon2Duplex.permute Poseidon2CanonicalConstants.selected
        (toDuplex state) := by
  apply duplexStateExt
  · funext lane
    change
      Poseidon2PermutationSound.permute
          (TranscriptMachine.laneNat state) lane.val % goldilocksP =
        Poseidon2Reference.referencePermutation
          Poseidon2CanonicalConstants.selected
          (fun inputLane => (state.lanes inputLane).val) lane
    have inputCanonical :
        ∀ inputLane, inputLane < TranscriptMachine.width →
          TranscriptMachine.laneNat state inputLane < goldilocksP := by
      intro inputLane inputLaneLt
      simp [TranscriptMachine.laneNat, inputLaneLt]
    rw [Poseidon2ExtractedReference.permute_eq_reference inputCanonical lane]
    have referenceEq :
        Poseidon2Reference.referencePermutation
            Poseidon2CanonicalConstants.selected
            (fun inputLane =>
              TranscriptMachine.laneNat state inputLane.val) lane =
          Poseidon2Reference.referencePermutation
            Poseidon2CanonicalConstants.selected
            (fun inputLane => (state.lanes inputLane).val) lane := by
      congr 2
      funext inputLane
      have inputLaneLt : inputLane.val < TranscriptMachine.width := by
        change inputLane.val < Poseidon2Core.width
        exact inputLane.isLt
      simp [TranscriptMachine.laneNat, inputLaneLt]
    rw [referenceEq]
    exact Nat.mod_eq_of_lt
      (Poseidon2Honest.refTerminal_lt Poseidon2CanonicalConstants.selected
        (fun inputLane => (state.lanes inputLane).val)
        Poseidon2Schedule.halfFullRounds lane)
  · rfl

/-- One canonical transcript-machine overwrite has the same value and cursor
as one duplex overwrite. -/
theorem absorbElem_toDuplex
    (state : TranscriptMachine.State) (value : TranscriptMachine.Field) :
    toDuplex (TranscriptMachine.absorbElem state value) =
      Poseidon2Duplex.absorbElem Poseidon2CanonicalConstants.selected
        value.val (toDuplex state) := by
  by_cases room : state.absorbed.val < TranscriptMachine.rate
  · have notFull :
        ¬Poseidon2Sponge.rate ≤ state.absorbed.val := by
      simpa [TranscriptMachine.rate, Poseidon2Sponge.rate] using
        (Nat.not_le_of_lt room)
    apply duplexStateExt
    · funext lane
      by_cases selected : lane.val = state.absorbed.val
      · simp [TranscriptMachine.absorbElem, room,
          TranscriptMachine.overwriteLane, Poseidon2Duplex.absorbElem,
          Poseidon2Duplex.guarded, notFull, toDuplex, selected,
          Nat.mod_eq_of_lt value.isLt]
      · simp [TranscriptMachine.absorbElem, room,
          TranscriptMachine.overwriteLane, Poseidon2Duplex.absorbElem,
          Poseidon2Duplex.guarded, notFull, toDuplex, selected,
          Nat.mod_eq_of_lt value.isLt]
    · simp [TranscriptMachine.absorbElem, room,
        Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded, notFull,
        toDuplex]
  · have full : Poseidon2Sponge.rate ≤ state.absorbed.val := by
      simpa [TranscriptMachine.rate, Poseidon2Sponge.rate] using
        (Nat.le_of_not_gt room)
    apply duplexStateExt
    · funext lane
      by_cases selected : lane.val = 0
      · simp [TranscriptMachine.absorbElem, room,
          TranscriptMachine.overwriteLane,
          Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded, full,
          Poseidon2Duplex.permute_absorbed, toDuplex, selected,
          Nat.mod_eq_of_lt value.isLt]
      · have laneEqual := congrArg (fun duplex => duplex.lanes lane)
          (permute_toDuplex state)
        simpa [TranscriptMachine.absorbElem, room,
          TranscriptMachine.overwriteLane,
          Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded, full,
          Poseidon2Duplex.permute_absorbed, toDuplex, selected,
          Nat.mod_eq_of_lt value.isLt] using laneEqual
    · simp [TranscriptMachine.absorbElem, room,
        Poseidon2Duplex.absorbElem, Poseidon2Duplex.guarded, full,
        Poseidon2Duplex.permute_absorbed, toDuplex]

/-- Transcript digest state transition is the independent duplex gate on the
same state. -/
theorem digest_state_toDuplex (state : TranscriptMachine.State) :
    toDuplex (TranscriptMachine.digest state).1 =
      Poseidon2Duplex.gate Poseidon2CanonicalConstants.selected
        (toDuplex state) := by
  unfold TranscriptMachine.digest Poseidon2Duplex.gate
  rw [permute_toDuplex, absorbElem_toDuplex]
  rfl

/-- Each transcript digest output is the matching lane of the independent
duplex gate. -/
theorem digest_output_toDuplex
    (state : TranscriptMachine.State) (lane : Fin 4) :
    ((TranscriptMachine.digest state).2 lane).val =
      (Poseidon2Duplex.gate Poseidon2CanonicalConstants.selected
        (toDuplex state)).lanes ⟨lane.val, by
          have laneLt := lane.isLt
          change lane.val < 8
          omega⟩ := by
  have stateEqual := congrArg
    (fun duplex => duplex.lanes ⟨lane.val, by
      have laneLt := lane.isLt
      change lane.val < 8
      omega⟩)
    (digest_state_toDuplex state)
  simpa [TranscriptMachine.digest, toDuplex] using stateEqual

/-- External-column execution preserves the independent duplex absorb-list
semantics on the exact values read from those columns. -/
theorem semanticExecute_external_toDuplex
    (assignment : Nat → Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (columns : List Nat) :
    toDuplex
        (ColumnReplay.semanticExecute assignment canonical run
          (columns.map ColumnReplay.Operation.external)).state =
      Poseidon2Duplex.absorbList Poseidon2CanonicalConstants.selected
        (columns.map assignment) (toDuplex run.state) := by
  induction columns generalizing run with
  | nil => rfl
  | cons column rest inductionHypothesis =>
      simp only [List.map_cons, ColumnReplay.semanticExecute,
        Poseidon2Duplex.absorbList]
      rw [inductionHypothesis]
      apply congrArg
        (Poseidon2Duplex.absorbList Poseidon2CanonicalConstants.selected
          (rest.map assignment))
      simpa [ColumnReplay.semanticStep] using
        absorbElem_toDuplex run.state
          (fieldAt assignment canonical column)

/-- Slice-end normalization is the duplex guard on the converted state. -/
theorem semanticNormalizeSlice_toDuplex (run : ColumnReplay.SemanticRun) :
    toDuplex (ColumnReplay.semanticNormalizeSlice run).state =
      Poseidon2Duplex.guarded Poseidon2CanonicalConstants.selected
        (toDuplex run.state) := by
  unfold ColumnReplay.semanticNormalizeSlice Poseidon2Duplex.guarded
  by_cases full :
      Poseidon2Sponge.rate ≤ (toDuplex run.state).absorbed
  · have machineFull :
        TranscriptMachine.rate ≤ run.state.absorbed.val := by
      simpa [toDuplex, TranscriptMachine.rate, Poseidon2Sponge.rate] using full
    simp only [machineFull, full, ↓reduceIte]
    exact permute_toDuplex run.state
  · have machineNotFull :
        ¬TranscriptMachine.rate ≤ run.state.absorbed.val := by
      simpa [toDuplex, TranscriptMachine.rate, Poseidon2Sponge.rate] using full
    simp only [machineNotFull, full, ↓reduceIte]

/-- A complete external-column slice is exactly the independent duplex bulk
absorb on the same assignment values. -/
theorem semanticExecuteSlice_external_toDuplex
    (assignment : Nat → Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (columns : List Nat) :
    toDuplex
        (ColumnReplay.semanticExecuteSlice assignment canonical run
          (columns.map ColumnReplay.Operation.external)).state =
      Poseidon2Duplex.absorbSlice Poseidon2CanonicalConstants.selected
        (columns.map assignment) (toDuplex run.state) := by
  unfold ColumnReplay.semanticExecuteSlice Poseidon2Duplex.absorbSlice
  rw [semanticNormalizeSlice_toDuplex,
    semanticExecute_external_toDuplex]

end Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
