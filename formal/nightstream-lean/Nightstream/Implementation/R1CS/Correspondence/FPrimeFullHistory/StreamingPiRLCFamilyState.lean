import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCompleteRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigest

/-!
Contract: exact interpretation of every generated PiRLC family-state column.

Owns the parity conversion, the complete 937-column decomposition, decoding
of one before and one after `FamilyState`, and the four absorbed-cursor
constant rows. It derives all source-row placement structures from one raw
body assignment.

Does not own the family arithmetic, overlay links, public cursor offset,
collision resistance, selective lowering, or recursive lifecycle
integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Concrete

abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

def parityForArm : ArmKind → CursorParity
  | .even => .even
  | .odd => .odd

def absorbedCursor (kind : ArmKind) : StateSide → Fin (rate + 1)
  | .before => beforeAbsorbed (parityForArm kind)
  | .after => afterAbsorbed (parityForArm kind)

def absorbedValue (kind : ArmKind) (side : StateSide) : Nat :=
  (absorbedCursor kind side).val

def stateColumns (kind : ArmKind) : StateSide → List Nat
  | .before => (armFor kind).beforeStateColumns
  | .after => (armFor kind).afterStateColumns

def replayStateColumns
    (kind : ArmKind) (side : StateSide) (replay : ReplayKind) : List Nat :=
  match side with
  | .before => beforeColumns (parityForArm kind) replay
  | .after => afterColumns (parityForArm kind) replay

def inputAbsorbedColumn (kind : ArmKind) (side : StateSide) : Nat :=
  (stateColumns kind side).getD 8 0

def outputAbsorbedColumn (kind : ArmKind) (side : StateSide) : Nat :=
  (stateColumns kind side).getD 125 0

def residualColumn (side : StateSide) (output : Fin 108) : Nat :=
  match side with
  | .before => sourceLayout.input.beforeResidual output
  | .after => sourceLayout.input.afterResidual output

def residualColumns (side : StateSide) : List Nat :=
  List.ofFn (residualColumn side)

def challengeColumn
    (side : StateSide) (source : Source) (lane : Fin ringDegree) : Nat :=
  match side with
  | .before => sourceLayout.beforeChallenge source lane
  | .after => sourceLayout.afterChallenge source lane

def challengeColumns (side : StateSide) : List Nat :=
  (List.ofFn fun source : Source =>
    List.ofFn fun lane : Fin ringDegree =>
      challengeColumn side source lane).flatten

def cursorColumn (side : StateSide) : Nat :=
  match side with
  | .before => sourceLayout.beforeCursor
  | .after => sourceLayout.afterCursor

/-- Semantic decomposition of the complete family-state serialization. The
two absorbed columns are local constant wires; every other column is shared
with the authoritative source relation. -/
def expectedStateColumns (kind : ArmKind) (side : StateSide) : List Nat :=
  replayStateColumns kind side .input ++
    [inputAbsorbedColumn kind side] ++
    residualColumns side ++
    replayStateColumns kind side .output ++
    [outputAbsorbedColumn kind side] ++
    challengeColumns side ++
    [cursorColumn side]

private theorem ofFn_affine_eq_range'
    (start count : Nat) :
    List.ofFn (fun index : Fin count => start + index.val) =
      List.range' start count := by
  induction count generalizing start with
  | zero => simp
  | succ count hypothesis =>
      rw [List.ofFn_succ, List.range'_succ]
      apply congrArg (start :: ·)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        hypothesis (start := start + 1)

private theorem range'_add (start left right : Nat) :
    List.range' start (left + right) =
      List.range' start left ++ List.range' (start + left) right := by
  apply (List.range'_eq_append_iff).2
  refine ⟨left, by omega, rfl, ?_⟩
  simp

private theorem flatten_affine_blocks_eq_range'
    (start blocks width : Nat) :
    (List.ofFn fun block : Fin blocks =>
      List.ofFn fun offset : Fin width =>
        start + block.val * width + offset.val).flatten =
      List.range' start (blocks * width) := by
  induction blocks generalizing start with
  | zero => simp
  | succ blocks hypothesis =>
      rw [List.ofFn_succ, List.flatten_cons]
      simp only [Fin.val_zero, Nat.zero_mul, Nat.add_zero]
      rw [ofFn_affine_eq_range']
      have tail :
          (List.ofFn fun block : Fin blocks =>
            List.ofFn fun offset : Fin width =>
              start + block.succ.val * width + offset.val).flatten =
            List.range' (start + width) (blocks * width) := by
        have functionsEqual :
            (fun block : Fin blocks =>
              List.ofFn fun offset : Fin width =>
                start + block.succ.val * width + offset.val) =
              (fun block : Fin blocks =>
                List.ofFn fun offset : Fin width =>
                  (start + width) + block.val * width + offset.val) := by
          funext block
          apply congrArg List.ofFn
          funext offset
          simp only [Fin.val_succ, Nat.add_mul, Nat.one_mul]
          omega
        rw [functionsEqual]
        exact hypothesis (start := start + width)
      rw [tail]
      rw [show Nat.succ blocks * width = width + blocks * width by
        simp [Nat.succ_mul, Nat.add_comm]]
      exact (range'_add start width (blocks * width)).symm

private theorem residual_columns_exact (side : StateSide) :
    residualColumns side =
      match side with
      | .before => List.range' 163610 108
      | .after => List.range' 163718 108 := by
  cases side with
  | before =>
      change List.ofFn (fun output : Fin 108 => 163610 + output.val) = _
      exact ofFn_affine_eq_range' 163610 108
  | after =>
      change List.ofFn (fun output : Fin 108 => 163718 + output.val) = _
      exact ofFn_affine_eq_range' 163718 108

private theorem challenge_columns_exact (side : StateSide) :
    challengeColumns side =
      match side with
      | .before => List.range' 163826 918
      | .after => List.range' 164744 918 := by
  cases side with
  | before =>
      change
        (List.ofFn fun source : Fin 17 =>
          List.ofFn fun lane : Fin 54 =>
            163826 + source.val * 54 + lane.val).flatten = _
      simpa using flatten_affine_blocks_eq_range' 163826 17 54
  | after =>
      change
        (List.ofFn fun source : Fin 17 =>
          List.ofFn fun lane : Fin 54 =>
            164744 + source.val * 54 + lane.val).flatten = _
      simpa using flatten_affine_blocks_eq_range' 164744 17 54

private def stateColumnSegments : ArmKind → StateSide → List StateColumnSegment
  | kind, .before => beforeStateColumnSegments kind
  | kind, .after => afterStateColumnSegments kind

private theorem state_columns_segments_exact
    (kind : ArmKind) (side : StateSide) :
    stateColumns kind side =
      expandSegments (stateColumnSegments kind side) := by
  cases side
  · simpa [stateColumns, stateColumnSegments] using
      exact_after_state_column_segments kind
  · simpa [stateColumns, stateColumnSegments] using
      exact_before_state_column_segments kind

private def expectedInputAbsorbed : ArmKind → StateSide → Nat
  | .even, .before => 311014
  | .even, .after => 311016
  | .odd, .before => 312214
  | .odd, .after => 312216

private def expectedOutputAbsorbed : ArmKind → StateSide → Nat
  | .even, .before => 311015
  | .even, .after => 311017
  | .odd, .before => 312215
  | .odd, .after => 312217

private theorem input_absorbed_column_exact
    (kind : ArmKind) (side : StateSide) :
    inputAbsorbedColumn kind side = expectedInputAbsorbed kind side := by
  cases kind <;> cases side <;>
    unfold inputAbsorbedColumn expectedInputAbsorbed <;>
    rw [state_columns_segments_exact] <;> rfl

private theorem output_absorbed_column_exact
    (kind : ArmKind) (side : StateSide) :
    outputAbsorbedColumn kind side = expectedOutputAbsorbed kind side := by
  cases kind <;> cases side <;>
    unfold outputAbsorbedColumn expectedOutputAbsorbed <;>
    rw [state_columns_segments_exact] <;> rfl

private theorem expected_state_columns_segments_exact
    (kind : ArmKind) (side : StateSide) :
    expectedStateColumns kind side =
      expandSegments (stateColumnSegments kind side) := by
  cases kind <;> cases side <;>
    unfold expectedStateColumns <;>
    rw [input_absorbed_column_exact, output_absorbed_column_exact,
      residual_columns_exact, challenge_columns_exact] <;>
    rfl

/-- Rust emits all 1,045 columns in the exact Lean `familyStateFields` order. -/
theorem state_columns_exact (kind : ArmKind) (side : StateSide) :
    stateColumns kind side = expectedStateColumns kind side := by
  exact (state_columns_segments_exact kind side).trans
    (expected_state_columns_segments_exact kind side).symm

/-! ## Absorbed-cursor constant rows -/

def absorbedPins (kind : ArmKind) (side : StateSide) : List (Nat × Nat) :=
  [(inputAbsorbedColumn kind side, absorbedValue kind side),
   (outputAbsorbedColumn kind side, absorbedValue kind side)]

def normalizedAbsorbedPinRows
    (kind : ArmKind) (side : StateSide) : List Row :=
  Poseidon2Normalized.normalizeProgram
    (ConstantPins.rows (absorbedPins kind side))

private theorem absorbed_pins_canonical
    (kind : ArmKind) (side : StateSide) :
    ConstantPins.ValuesCanonical (absorbedPins kind side) := by
  cases kind <;> cases side <;> native_decide

private theorem absorbed_pins_in_glue
    (kind : ArmKind) (side : StateSide) :
    rowsIncluded (normalizedAbsorbedPinRows kind side)
      (glueProgram kind) = true := by
  cases kind <;> cases side <;> native_decide

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem absorbed_pin_facts
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    ∀ pin ∈ absorbedPins kind side,
      assignment pin.1 = pin.2 := by
  have normalizedSatisfies :
      Satisfies (normalizedAbsorbedPinRows kind side) assignment := by
    intro row member
    exact glue_satisfies kind assignment satisfied row
      (rowsIncluded_sound (absorbed_pins_in_glue kind side) row member)
  have pinRowsSatisfy :
      Satisfies (ConstantPins.rows (absorbedPins kind side)) assignment :=
    (Poseidon2Normalized.satisfies_normalizeProgram
      (ConstantPins.rows (absorbedPins kind side)) assignment).mp
        normalizedSatisfies
  exact ConstantPins.sound
    (programRows := ConstantPins.rows (absorbedPins kind side))
    (absorbed_pins_canonical kind side)
    (by cases kind <;> cases side <;> native_decide)
    canonical one pinRowsSatisfy

theorem input_absorbed_exact
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (inputAbsorbedColumn kind side) =
      absorbedValue kind side := by
  exact absorbed_pin_facts kind side assignment canonical one satisfied
    (inputAbsorbedColumn kind side, absorbedValue kind side)
    (by simp [absorbedPins])

theorem output_absorbed_exact
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (outputAbsorbedColumn kind side) =
      absorbedValue kind side := by
  exact absorbed_pin_facts kind side assignment canonical one satisfied
    (outputAbsorbedColumn kind side, absorbedValue kind side)
    (by simp [absorbedPins])

/-! ## Exact semantic decoding -/

def assignmentField
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) : Nightstream.SuperNeo.Concrete.F :=
  ⟨assignment column, by
    simpa [goldilocksP, goldilocksModulus] using canonical column⟩

@[simp] theorem assignmentField_val
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    (assignmentField assignment canonical column).val = assignment column :=
  rfl

def bindingStateAt
    (assignment : Nat → Nat) (kind : ArmKind)
    (side : StateSide) (replay : ReplayKind) : BindingState :=
  stateAt assignment (replayStateColumns kind side replay)
    (absorbedCursor kind side)

def residualAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (side : StateSide) : InputResidual :=
  fun output => assignmentField assignment canonical (residualColumn side output)

def challengesAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (side : StateSide) : Source → RingF :=
  fun source lane =>
    assignmentField assignment canonical (challengeColumn side source lane)

/-- The unique semantic family state decoded from one physical side. -/
def familyStateAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (kind : ArmKind) (side : StateSide) : FamilyState where
  inputReplay := bindingStateAt assignment kind side .input
  inputResidual := residualAt assignment canonical side
  outputReplay := bindingStateAt assignment kind side .output
  challenges := challengesAt assignment canonical side
  familyCursor := assignment (cursorColumn side)

private theorem binding_fields_exact
    (assignment : Nat → Nat)
    (kind : ArmKind) (side : StateSide) (replay : ReplayKind) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.bindingFields
        (bindingStateAt assignment kind side replay) =
      (replayStateColumns kind side replay).map assignment ++
        [absorbedValue kind side] := by
  cases kind <;> cases side <;> cases replay <;>
    simp [Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.bindingFields,
      bindingStateAt, replayStateColumns, parityForArm,
      absorbedCursor, absorbedValue, stateAt, beforeColumns, afterColumns, arm,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.evenArm,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.oddArm]
  all_goals congr 1

private theorem residual_fields_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (side : StateSide) :
    inputResidualFields (residualAt assignment canonical side) =
      (residualColumns side).map assignment := by
  unfold inputResidualFields residualAt residualColumns
  rw [List.map_ofFn]
  rfl

private theorem challenge_fields_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (side : StateSide) :
    phaseFields (challengesAt assignment canonical side) =
      (challengeColumns side).map assignment := by
  rw [challengeColumns, List.map_flatten, phaseFields, sourceBlocks]
  simp only [List.map_ofFn, ringFields, challengesAt, assignmentField]
  congr 1

/-- The digest preimage is exactly the canonical serialization of the
decoded `FamilyState`; no state field remains independent. -/
theorem family_state_fields_exact
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (stateColumns kind side).map assignment =
      familyStateFields (familyStateAt assignment canonical kind side) := by
  rw [state_columns_exact]
  simp only [expectedStateColumns, List.map_append, List.map_cons,
    List.map_nil, familyStateFields, familyStateAt]
  rw [binding_fields_exact assignment kind side .input,
    residual_fields_exact assignment canonical side,
    binding_fields_exact assignment kind side .output,
    challenge_fields_exact assignment canonical side,
    input_absorbed_exact kind side assignment canonical one satisfied,
    output_absorbed_exact kind side assignment canonical one satisfied]
  simp [List.append_assoc]

private theorem map_getD_range {alpha : Type}
    (entries : List alpha) (fallback : alpha) :
    (List.range entries.length).map
        (fun index => entries.getD index fallback) =
      entries := by
  induction entries with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      exact congrArg (head :: ·) hypothesis

private theorem state_column_count
    (kind : ArmKind) (side : StateSide) :
    (stateColumns kind side).length = 1045 := by
  cases side with
  | before => exact (exact_state_column_shape kind).1
  | after => exact (exact_state_column_shape kind).2.1

private theorem state_word_column_exact
    (kind : ArmKind) (side : StateSide) (index : Nat) :
    stateWordColumnFor kind side index =
      (stateColumns kind side).getD index 0 := by
  cases side <;> rfl

/-- The 1,045 external operations in one digest path read the exact canonical
serialization of the decoded semantic family state. -/
theorem digest_preimage_is_family_state
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (List.range 1045).map (fun index =>
        assignment (stateWordColumnFor kind side index)) =
      familyStateFields
        (familyStateAt assignment canonical kind side) := by
  rw [← state_column_count kind side]
  simp_rw [state_word_column_exact]
  calc
    (List.range (stateColumns kind side).length).map (fun index =>
        assignment ((stateColumns kind side).getD index 0)) =
        (stateColumns kind side).map assignment := by
      have exactColumns := congrArg (List.map assignment)
        (map_getD_range (stateColumns kind side) 0)
      simpa [List.map_map, Function.comp_def] using exactColumns
    _ = familyStateFields
          (familyStateAt assignment canonical kind side) :=
      family_state_fields_exact kind side assignment canonical one satisfied

/-- The decoded before and after states occupy the exact replay state
columns, including the physical cursor parity. -/
theorem replay_states_placed
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    FamilyStatesPlaced (parityForArm kind) assignment
      (familyStateAt assignment canonical kind .before)
      (familyStateAt assignment canonical kind .after) := by
  constructor <;>
    simp [familyStateAt, bindingStateAt, replayStateColumns, absorbedCursor]

/-- The decoded before and after residuals occupy the exact source-layout
columns. -/
theorem residuals_placed
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
      sourceLayout.input assignment
      (familyStateAt assignment canonical kind .before).inputResidual
      (familyStateAt assignment canonical kind .after).inputResidual := by
  constructor <;> intro output <;> rfl

/-- The decoded challenge vectors and family cursors occupy the exact
source-layout columns. -/
theorem carry_state_placed
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
        sourceLayout)
      assignment
      (familyStateAt assignment canonical kind .before)
      (familyStateAt assignment canonical kind .after) := by
  simp [
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced,
    familyStateAt, challengesAt, challengeColumn, cursorColumn,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState
