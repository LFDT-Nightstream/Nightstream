import Mathlib.Tactic.IntervalCases
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.ExtractedReference

/-!
Contract: same-assignment semantic sequence for the recursive-terminal XOut
public Poseidon2 calls.

Owns the generic final-row-to-permutation bridge and the first concrete call
step. Later call steps must reuse these definitions and one small artifact
certificate per call.

Does not own the complete terminal matrix, final relation satisfaction,
public-word copies, lifecycle composition, or collision resistance.

Assurance tier: artifact-checked once the explicit final-row premises hold.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallArtifact
open Nightstream.Implementation.R1CS.Poseidon2Sponge

def callInputValues
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Nat) : Nat :=
  if bounded : lane < width then
    (inputValue placement assignment ⟨lane, bounded⟩).val
  else
    0

@[simp] theorem callInputValues_fin
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) :
    callInputValues placement assignment lane.val =
      (inputValue placement assignment lane).val := by
  simp [callInputValues, lane.isLt]

def callOutputValue
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) : Nat :=
  lcEval
    (sourcePhysical
      (reconstructedSource (projectFinalAssignment placement assignment)))
    (traceFinalForm lane)

def callOutputState
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Nat) : Nat :=
  if bounded : lane < width then
    callOutputValue placement assignment ⟨lane, bounded⟩
  else
    0

@[simp] theorem callOutputState_fin
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) :
    callOutputState placement assignment lane.val =
      callOutputValue placement assignment lane := by
  simp [callOutputState, lane.isLt]

/-- The selected 86-row slice computes the executable permutation extracted
from the exact 600-row Rust call, for any compact call placement. -/
theorem final_rows_compute_permute
    {placement : PoseidonCallPlacement} {valid : placement.Valid}
    {rows : Nat}
    {relation : InterpretedRelation rows placement.finalColumns}
    {assignment : AbsoluteAssignment placement}
    (exact : FinalRowSliceExact placement valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment placement.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue placement assignment lane =
      Poseidon2PermutationSound.permute
        (callInputValues placement assignment) lane.val := by
  have canonical : ∀ inputLane, inputLane < 8 →
      callInputValues placement assignment inputLane < goldilocksP := by
    intro inputLane inputLaneLt
    have bounded : inputLane < width := by simpa only [width] using inputLaneLt
    simp only [callInputValues, dif_pos bounded]
    simpa [goldilocksP, goldilocksModulus] using
      (inputValue placement assignment ⟨inputLane, bounded⟩).isLt
  calc
    callOutputValue placement assignment lane =
        referencePermutation Poseidon2CanonicalConstants.selected
          (fun inputLane =>
            (inputValue placement assignment inputLane).val) lane :=
      final_rows_compute_reference exact satisfied one selectorOne lane
    _ = referencePermutation Poseidon2CanonicalConstants.selected
          (fun inputLane =>
            callInputValues placement assignment inputLane.val) lane := by
      congr 2
      funext inputLane
      exact (callInputValues_fin placement assignment inputLane).symm
    _ = Poseidon2PermutationSound.permute
          (callInputValues placement assignment) lane.val :=
      (Poseidon2ExtractedReference.permute_eq_reference canonical lane).symm

private theorem valueRound_state_congr
    (round : Round) (values : List Nat)
    (leftState rightState : Nat → Nat)
    (same : ∀ lane, lane < 8 → leftState lane = rightState lane) :
    valueRound round values leftState =
      valueRound round values rightState := by
  funext outputLane
  unfold valueRound
  apply Poseidon2PermutationSound.permute_congr
  intro lane laneLt
  cases kind : round.kind with
  | absorb columns =>
      simp only [valueInput, kind]
      split
      · rw [same lane laneLt]
      · exact same lane laneLt
  | pad =>
      simp only [valueInput, kind]
      split
      · rw [same 0 (by decide)]
      · exact same lane laneLt

def firstChunk
    (assignment : AbsoluteAssignment callPlacement0) : List Nat :=
  [ (xOutValue assignment 0).val
  , (xOutValue assignment 1).val
  , (xOutValue assignment 2).val
  , (xOutValue assignment 3).val ]

theorem callPlacement0_inputs_match
    (assignment : AbsoluteAssignment callPlacement0)
    (lane : Nat) (laneLt : lane < 8) :
    callInputValues callPlacement0 assignment lane =
      valueInput (rounds.getD 0 default).kind (firstChunk assignment)
        (fun _ => 0) lane := by
  have bounded : lane < width := by simpa only [width] using laneLt
  rw [show callInputValues callPlacement0 assignment lane =
      (inputValue callPlacement0 assignment ⟨lane, bounded⟩).val by
    exact callInputValues_fin callPlacement0 assignment ⟨lane, bounded⟩]
  rw [callPlacement0_input_values assignment ⟨lane, bounded⟩]
  rw [round0_kind]
  interval_cases lane <;>
    simp [valueInput, firstChunk, goldilocksP,
      goldilocksModulus, Nat.mod_eq_of_lt]

/-- One concrete generated call now refines one exact pure sponge round. -/
theorem callPlacement0_final_rows_compute_first_round
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement0.finalColumns}
    {assignment : AbsoluteAssignment callPlacement0}
    (exact : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement0.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement0 assignment lane =
      valueRound (rounds.getD 0 default) (firstChunk assignment)
        (fun _ => 0) lane.val := by
  rw [final_rows_compute_permute exact satisfied one selectorOne lane]
  unfold valueRound
  apply Poseidon2PermutationSound.permute_congr
  intro inputLane inputLaneLt
  exact callPlacement0_inputs_match assignment inputLane inputLaneLt

def secondChunk
    (assignment : AbsoluteAssignment callPlacement0) : List Nat :=
  [ (xOutValue assignment 4).val
  , (xOutValue assignment 5).val
  , (xOutValue assignment 6).val
  , (xOutValue assignment 7).val ]

def call0State
    (assignment : AbsoluteAssignment callPlacement0) (lane : Nat) : Nat :=
  if bounded : lane < width then
    callOutputValue callPlacement0 assignment ⟨lane, bounded⟩
  else
    0

@[simp] theorem call0State_fin
    (assignment : AbsoluteAssignment callPlacement0) (lane : Fin width) :
    call0State assignment lane.val =
      callOutputValue callPlacement0 assignment lane := by
  simp [call0State, lane.isLt]

private theorem call0_output_port_value
    (assignment : AbsoluteAssignment callPlacement0) (lane : Fin width) :
    (absolutePortAction assignment
      (callOutputPort callPlacement0 lane)).val =
        callOutputValue callPlacement0 assignment lane := by
  exact callOutputPort_action callPlacement0_valid assignment lane

theorem callPlacement1_input_values
    (assignment : AbsoluteAssignment callPlacement0) (lane : Fin width) :
    inputValue callPlacement1 assignment lane =
      if lane.val < 4 then
        absolutePortAction assignment (callOutputPort callPlacement0 lane) +
          xOutValue assignment (4 + lane.val)
      else
        absolutePortAction assignment (callOutputPort callPlacement0 lane) := by
  unfold inputValue
  rw [callPlacement1_input_port_exact]
  split
  · rw [absolutePortAction_append]
    rfl
  · rfl

theorem callPlacement1_inputs_match
    (assignment : AbsoluteAssignment callPlacement0)
    (lane : Nat) (laneLt : lane < 8) :
    callInputValues callPlacement1 assignment lane =
      valueInput (rounds.getD 1 default).kind (secondChunk assignment)
        (call0State assignment) lane := by
  have bounded : lane < width := by simpa only [width] using laneLt
  rw [show callInputValues callPlacement1 assignment lane =
      (inputValue callPlacement1 assignment ⟨lane, bounded⟩).val by
    exact callInputValues_fin callPlacement1 assignment ⟨lane, bounded⟩]
  rw [callPlacement1_input_values assignment ⟨lane, bounded⟩]
  rw [round1_kind]
  simp only [valueInput]
  rw [show call0State assignment lane =
      callOutputValue callPlacement0 assignment ⟨lane, bounded⟩ by
    exact call0State_fin assignment ⟨lane, bounded⟩]
  interval_cases lane <;>
    simp [secondChunk,
      call0_output_port_value, Fin.val_add, goldilocksP,
      goldilocksModulus]

/-- The exact call-1 slice computes the second pure sponge round from the
complete call-0 compact output on the same assignment. -/
theorem callPlacement1_final_rows_compute_second_round
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement1.finalColumns}
    {assignment : AbsoluteAssignment callPlacement1}
    (exact : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement1.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement1 assignment lane =
      valueRound (rounds.getD 1 default) (secondChunk assignment)
        (call0State assignment) lane.val := by
  rw [final_rows_compute_permute exact satisfied one selectorOne lane]
  unfold valueRound
  apply Poseidon2PermutationSound.permute_congr
  intro inputLane inputLaneLt
  exact callPlacement1_inputs_match assignment inputLane inputLaneLt

def xOutValueAt {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (index : Nat) : F :=
  absolutePortAction assignment
    (xOutImages.getD index emptySourceImage).port

def xOutChunkAt {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (offset : Nat) : List Nat :=
  [ (xOutValueAt assignment offset).val
  , (xOutValueAt assignment (offset + 1)).val
  , (xOutValueAt assignment (offset + 2)).val
  , (xOutValueAt assignment (offset + 3)).val ]

def priorPortState {current : PoseidonCallPlacement}
    (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current) (lane : Nat) : Nat :=
  if bounded : lane < width then
    (absolutePortAction assignment
      (callOutputPort previous ⟨lane, bounded⟩)).val
  else
    0

@[simp] theorem priorPortState_fin
    {current : PoseidonCallPlacement} (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current) (lane : Fin width) :
    priorPortState previous assignment lane.val =
      (absolutePortAction assignment (callOutputPort previous lane)).val := by
  simp [priorPortState, lane.isLt]

private theorem priorPortState_agrees_callOutputState
    {previous : PoseidonCallPlacement}
    (previousValid : previous.Valid)
    (assignment : AbsoluteAssignment previous)
    (lane : Nat) (laneLt : lane < 8) :
    priorPortState previous assignment lane =
      callOutputState previous assignment lane := by
  have bounded : lane < width := by simpa only [width] using laneLt
  calc
    priorPortState previous assignment lane =
        (absolutePortAction assignment
          (callOutputPort previous ⟨lane, bounded⟩)).val :=
      priorPortState_fin previous assignment ⟨lane, bounded⟩
    _ = callOutputValue previous assignment ⟨lane, bounded⟩ :=
      callOutputPort_action previousValid assignment ⟨lane, bounded⟩
    _ = callOutputState previous assignment lane :=
      (callOutputState_fin previous assignment ⟨lane, bounded⟩).symm

private theorem call_output_refines_round
    {current previous : PoseidonCallPlacement}
    (currentAssignment : AbsoluteAssignment current)
    (previousAssignment : AbsoluteAssignment previous)
    (round : Round) (values : List Nat)
    (sourceState targetState : Nat → Nat)
    (computed : ∀ lane : Fin width,
      callOutputValue current currentAssignment lane =
        valueRound round values sourceState lane.val)
    (sourceAgrees : ∀ lane, lane < 8 →
      sourceState lane = callOutputState previous previousAssignment lane)
    (previousAgrees : ∀ lane, lane < 8 →
      callOutputState previous previousAssignment lane = targetState lane) :
    ∀ lane, lane < 8 →
      callOutputState current currentAssignment lane =
        valueRound round values targetState lane := by
  intro lane laneLt
  have bounded : lane < width := by simpa only [width] using laneLt
  calc
    callOutputState current currentAssignment lane =
        callOutputValue current currentAssignment ⟨lane, bounded⟩ :=
      callOutputState_fin current currentAssignment ⟨lane, bounded⟩
    _ = valueRound round values sourceState lane :=
      computed ⟨lane, bounded⟩
    _ = valueRound round values targetState lane :=
      congrFun
        (valueRound_state_congr round values sourceState targetState
          (fun inputLane inputLaneLt =>
            (sourceAgrees inputLane inputLaneLt).trans
              (previousAgrees inputLane inputLaneLt))) lane

private theorem absorb_input_values
    {current : PoseidonCallPlacement} (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current) (offset : Nat)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port =
        absorbInputPort previous offset lane)
    (lane : Fin width) :
    inputValue current assignment lane =
      if lane.val < 4 then
        absolutePortAction assignment (callOutputPort previous lane) +
          xOutValueAt assignment (offset + lane.val)
      else
        absolutePortAction assignment (callOutputPort previous lane) := by
  unfold inputValue
  rw [portExact lane]
  unfold absorbInputPort
  split
  · rw [absolutePortAction_append]
    rfl
  · rfl

private theorem absorb_inputs_match
    {current : PoseidonCallPlacement} (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current) (offset : Nat)
    (columns : List Nat) (columnsLength : columns.length = 4)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port =
        absorbInputPort previous offset lane)
    (lane : Nat) (laneLt : lane < 8) :
    callInputValues current assignment lane =
      valueInput (.absorb columns) (xOutChunkAt assignment offset)
        (priorPortState previous assignment) lane := by
  have bounded : lane < width := by simpa only [width] using laneLt
  rw [show callInputValues current assignment lane =
      (inputValue current assignment ⟨lane, bounded⟩).val by
    exact callInputValues_fin current assignment ⟨lane, bounded⟩]
  rw [absorb_input_values previous assignment offset portExact
    ⟨lane, bounded⟩]
  simp only [valueInput]
  rw [columnsLength]
  rw [show priorPortState previous assignment lane =
      (absolutePortAction assignment
        (callOutputPort previous ⟨lane, bounded⟩)).val by
    exact priorPortState_fin previous assignment ⟨lane, bounded⟩]
  interval_cases lane <;>
    simp [xOutChunkAt, Fin.val_add, goldilocksP, goldilocksModulus]

private theorem absorb_final_rows_compute_round
    {current previous : PoseidonCallPlacement}
    {valid : current.Valid} {rows : Nat}
    {relation : InterpretedRelation rows current.finalColumns}
    {assignment : AbsoluteAssignment current}
    {round : Round} {columns : List Nat} (offset : Nat)
    (kindExact : round.kind = .absorb columns)
    (columnsLength : columns.length = 4)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port =
        absorbInputPort previous offset lane)
    (exact : FinalRowSliceExact current valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment current.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue current assignment lane =
      valueRound round (xOutChunkAt assignment offset)
        (priorPortState previous assignment) lane.val := by
  rw [final_rows_compute_permute exact satisfied one selectorOne lane]
  unfold valueRound
  apply Poseidon2PermutationSound.permute_congr
  intro inputLane inputLaneLt
  rw [kindExact]
  exact absorb_inputs_match previous assignment offset columns columnsLength
    portExact inputLane inputLaneLt

private theorem onePort_action
    {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement)
    (one : absoluteValue assignment 0 = 1) :
    absolutePortAction assignment onePort = 1 := by
  have fieldOne : fieldValue 1 = (1 : F) := by
    rw [fieldValue_of_lt 1 (by decide)]
    apply Fin.ext
    exact (Nat.mod_eq_of_lt (by decide)).symm
  simp [absolutePortAction, onePort, explicitAction,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.sum,
    fieldOne, one]

private theorem pad_input_values
    {current : PoseidonCallPlacement} (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port = padInputPort previous lane)
    (lane : Fin width) :
    inputValue current assignment lane =
      if lane.val = 0 then
        absolutePortAction assignment (callOutputPort previous lane) +
          absolutePortAction assignment onePort
      else
        absolutePortAction assignment (callOutputPort previous lane) := by
  unfold inputValue
  rw [portExact lane]
  unfold padInputPort
  split
  · rw [absolutePortAction_append]
  · rfl

private theorem pad_inputs_match
    {current : PoseidonCallPlacement} (previous : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment current)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port = padInputPort previous lane)
    (one : absoluteValue assignment 0 = 1)
    (lane : Nat) (laneLt : lane < 8) :
    callInputValues current assignment lane =
      valueInput .pad [] (priorPortState previous assignment) lane := by
  have bounded : lane < width := by simpa only [width] using laneLt
  rw [show callInputValues current assignment lane =
      (inputValue current assignment ⟨lane, bounded⟩).val by
    exact callInputValues_fin current assignment ⟨lane, bounded⟩]
  rw [pad_input_values previous assignment portExact ⟨lane, bounded⟩]
  simp only [valueInput]
  rw [onePort_action assignment one]
  interval_cases lane <;>
    simp [priorPortState, width, Fin.val_add, goldilocksP,
      goldilocksModulus]

private theorem pad_final_rows_compute_round
    {current previous : PoseidonCallPlacement}
    {valid : current.Valid} {rows : Nat}
    {relation : InterpretedRelation rows current.finalColumns}
    {assignment : AbsoluteAssignment current}
    {round : Round} (kindExact : round.kind = .pad)
    (portExact : ∀ lane : Fin width,
      (inputImage current lane).port = padInputPort previous lane)
    (exact : FinalRowSliceExact current valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment current.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue current assignment lane =
      valueRound round [] (priorPortState previous assignment) lane.val := by
  rw [final_rows_compute_permute exact satisfied one selectorOne lane]
  unfold valueRound
  apply Poseidon2PermutationSound.permute_congr
  intro inputLane inputLaneLt
  rw [kindExact]
  exact pad_inputs_match previous assignment portExact one
    inputLane inputLaneLt

theorem callPlacement2_final_rows_compute_round2
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement2.finalColumns}
    {assignment : AbsoluteAssignment callPlacement2}
    (exact : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement2.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement2 assignment lane =
      valueRound (rounds.getD 2 default) (xOutChunkAt assignment 8)
        (priorPortState callPlacement1 assignment) lane.val := by
  exact absorb_final_rows_compute_round 8 round2_kind rfl
    callPlacement2_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement3_final_rows_compute_round3
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement3.finalColumns}
    {assignment : AbsoluteAssignment callPlacement3}
    (exact : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement3.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement3 assignment lane =
      valueRound (rounds.getD 3 default) (xOutChunkAt assignment 12)
        (priorPortState callPlacement2 assignment) lane.val := by
  exact absorb_final_rows_compute_round 12 round3_kind rfl
    callPlacement3_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement4_final_rows_compute_round4
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement4.finalColumns}
    {assignment : AbsoluteAssignment callPlacement4}
    (exact : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement4.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement4 assignment lane =
      valueRound (rounds.getD 4 default) (xOutChunkAt assignment 16)
        (priorPortState callPlacement3 assignment) lane.val := by
  exact absorb_final_rows_compute_round 16 round4_kind rfl
    callPlacement4_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement5_final_rows_compute_round5
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement5.finalColumns}
    {assignment : AbsoluteAssignment callPlacement5}
    (exact : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement5.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement5 assignment lane =
      valueRound (rounds.getD 5 default) (xOutChunkAt assignment 20)
        (priorPortState callPlacement4 assignment) lane.val := by
  exact absorb_final_rows_compute_round 20 round5_kind rfl
    callPlacement5_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement6_final_rows_compute_round6
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement6.finalColumns}
    {assignment : AbsoluteAssignment callPlacement6}
    (exact : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement6.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement6 assignment lane =
      valueRound (rounds.getD 6 default) (xOutChunkAt assignment 24)
        (priorPortState callPlacement5 assignment) lane.val := by
  exact absorb_final_rows_compute_round 24 round6_kind rfl
    callPlacement6_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement7_final_rows_compute_round7
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement7.finalColumns}
    {assignment : AbsoluteAssignment callPlacement7}
    (exact : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement7.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement7 assignment lane =
      valueRound (rounds.getD 7 default) (xOutChunkAt assignment 28)
        (priorPortState callPlacement6 assignment) lane.val := by
  exact absorb_final_rows_compute_round 28 round7_kind rfl
    callPlacement7_input_port_exact exact satisfied one selectorOne lane

theorem callPlacement8_final_rows_compute_pad
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (exact : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement8 assignment lane =
      valueRound (rounds.getD 8 default) []
        (priorPortState callPlacement7 assignment) lane.val := by
  exact pad_final_rows_compute_round round8_kind
    callPlacement8_input_port_exact exact satisfied one selectorOne lane

def terminalXOutValues
    (assignment : AbsoluteAssignment callPlacement8) : List Nat :=
  firstChunk assignment ++
    secondChunk assignment ++
    xOutChunkAt assignment 8 ++
    xOutChunkAt assignment 12 ++
    xOutChunkAt assignment 16 ++
    xOutChunkAt assignment 20 ++
    xOutChunkAt assignment 24 ++
    xOutChunkAt assignment 28

def terminalState0
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 0 default) (firstChunk assignment) (fun _ => 0)

def terminalState1
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 1 default) (secondChunk assignment)
    (terminalState0 assignment)

def terminalState2
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 2 default) (xOutChunkAt assignment 8)
    (terminalState1 assignment)

def terminalState3
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 3 default) (xOutChunkAt assignment 12)
    (terminalState2 assignment)

def terminalState4
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 4 default) (xOutChunkAt assignment 16)
    (terminalState3 assignment)

def terminalState5
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 5 default) (xOutChunkAt assignment 20)
    (terminalState4 assignment)

def terminalState6
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 6 default) (xOutChunkAt assignment 24)
    (terminalState5 assignment)

def terminalState7
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 7 default) (xOutChunkAt assignment 28)
    (terminalState6 assignment)

def terminalState8
    (assignment : AbsoluteAssignment callPlacement8) : Nat → Nat :=
  valueRound (rounds.getD 8 default) [] (terminalState7 assignment)

/-- Fixed nine-round geometry: the eight four-field chunks followed by the
terminal pad are exactly the generated pure sponge schedule. -/
private theorem terminalState8_eq_runValueRounds
    (assignment : AbsoluteAssignment callPlacement8) :
    terminalState8 assignment =
      runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0) := by
  rfl

/-- The nine exact Rust final-row slices compute the pure Poseidon2 hash of
the ordered 32-field XOut frame on one final selective-CCS assignment. -/
theorem final_rows_compute_terminal_x_out_hash
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (exact0 : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      assignment)
    (exact1 : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      assignment)
    (exact2 : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      assignment)
    (exact3 : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      assignment)
    (exact4 : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      assignment)
    (exact5 : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      assignment)
    (exact6 : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      assignment)
    (exact7 : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      assignment)
    (exact8 : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1)
    (lane : Fin width) :
    callOutputValue callPlacement8 assignment lane =
      runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0)
        lane.val := by
  have state0 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement0 assignment inputLane =
        terminalState0 assignment inputLane := by
    intro inputLane inputLaneLt
    have bounded : inputLane < width := by
      simpa only [width] using inputLaneLt
    calc
      callOutputState callPlacement0 assignment inputLane =
          callOutputValue callPlacement0 assignment ⟨inputLane, bounded⟩ :=
        callOutputState_fin callPlacement0 assignment ⟨inputLane, bounded⟩
      _ = terminalState0 assignment inputLane := by
        simpa only [terminalState0] using
          callPlacement0_final_rows_compute_first_round exact0 satisfied one
            selectorOne ⟨inputLane, bounded⟩
  have state1 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement1 assignment inputLane =
        terminalState1 assignment inputLane := by
    simpa only [terminalState1] using
      (call_output_refines_round
        (current := callPlacement1) (previous := callPlacement0)
        assignment assignment (rounds.getD 1 default) (secondChunk assignment)
        (call0State assignment) (terminalState0 assignment)
        (fun outputLane =>
          callPlacement1_final_rows_compute_second_round exact1 satisfied
            one selectorOne outputLane)
        (by
          intro inputLane inputLaneLt
          have bounded : inputLane < width := by
            simpa only [width] using inputLaneLt
          exact (call0State_fin assignment ⟨inputLane, bounded⟩).trans
            (callOutputState_fin callPlacement0 assignment
              ⟨inputLane, bounded⟩).symm)
        state0)
  have state2 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement2 assignment inputLane =
        terminalState2 assignment inputLane := by
    simpa only [terminalState2] using
      (call_output_refines_round
        (current := callPlacement2) (previous := callPlacement1)
        assignment assignment (rounds.getD 2 default) (xOutChunkAt assignment 8)
        (priorPortState callPlacement1 assignment) (terminalState1 assignment)
        (fun outputLane =>
          callPlacement2_final_rows_compute_round2 exact2 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement1)
          callPlacement1_valid assignment)
        state1)
  have state3 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement3 assignment inputLane =
        terminalState3 assignment inputLane := by
    simpa only [terminalState3] using
      (call_output_refines_round
        (current := callPlacement3) (previous := callPlacement2)
        assignment assignment (rounds.getD 3 default) (xOutChunkAt assignment 12)
        (priorPortState callPlacement2 assignment) (terminalState2 assignment)
        (fun outputLane =>
          callPlacement3_final_rows_compute_round3 exact3 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement2)
          callPlacement2_valid assignment)
        state2)
  have state4 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement4 assignment inputLane =
        terminalState4 assignment inputLane := by
    simpa only [terminalState4] using
      (call_output_refines_round
        (current := callPlacement4) (previous := callPlacement3)
        assignment assignment (rounds.getD 4 default) (xOutChunkAt assignment 16)
        (priorPortState callPlacement3 assignment) (terminalState3 assignment)
        (fun outputLane =>
          callPlacement4_final_rows_compute_round4 exact4 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement3)
          callPlacement3_valid assignment)
        state3)
  have state5 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement5 assignment inputLane =
        terminalState5 assignment inputLane := by
    simpa only [terminalState5] using
      (call_output_refines_round
        (current := callPlacement5) (previous := callPlacement4)
        assignment assignment (rounds.getD 5 default) (xOutChunkAt assignment 20)
        (priorPortState callPlacement4 assignment) (terminalState4 assignment)
        (fun outputLane =>
          callPlacement5_final_rows_compute_round5 exact5 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement4)
          callPlacement4_valid assignment)
        state4)
  have state6 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement6 assignment inputLane =
        terminalState6 assignment inputLane := by
    simpa only [terminalState6] using
      (call_output_refines_round
        (current := callPlacement6) (previous := callPlacement5)
        assignment assignment (rounds.getD 6 default) (xOutChunkAt assignment 24)
        (priorPortState callPlacement5 assignment) (terminalState5 assignment)
        (fun outputLane =>
          callPlacement6_final_rows_compute_round6 exact6 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement5)
          callPlacement5_valid assignment)
        state5)
  have state7 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement7 assignment inputLane =
        terminalState7 assignment inputLane := by
    simpa only [terminalState7] using
      (call_output_refines_round
        (current := callPlacement7) (previous := callPlacement6)
        assignment assignment (rounds.getD 7 default) (xOutChunkAt assignment 28)
        (priorPortState callPlacement6 assignment) (terminalState6 assignment)
        (fun outputLane =>
          callPlacement7_final_rows_compute_round7 exact7 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement6)
          callPlacement6_valid assignment)
        state6)
  have state8 : ∀ inputLane, inputLane < 8 →
      callOutputState callPlacement8 assignment inputLane =
        terminalState8 assignment inputLane := by
    simpa only [terminalState8] using
      (call_output_refines_round
        (current := callPlacement8) (previous := callPlacement7)
        assignment assignment (rounds.getD 8 default) []
        (priorPortState callPlacement7 assignment) (terminalState7 assignment)
        (fun outputLane =>
          callPlacement8_final_rows_compute_pad exact8 satisfied one
            selectorOne outputLane)
        (priorPortState_agrees_callOutputState
          (previous := callPlacement7)
          callPlacement7_valid assignment)
        state7)
  calc
    callOutputValue callPlacement8 assignment lane =
        callOutputState callPlacement8 assignment lane.val :=
      (callOutputState_fin callPlacement8 assignment lane).symm
    _ = terminalState8 assignment lane.val :=
      state8 lane.val (by simpa only [width] using lane.isLt)
    _ = runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0)
          lane.val :=
      congrFun (terminalState8_eq_runValueRounds assignment) lane.val

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
