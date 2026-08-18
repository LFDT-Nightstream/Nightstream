import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeStateDigestSliceCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeStateDigest

/-!
Contract: exact compact Rust Prelude rows imply the typed Prelude relation.

Owns the source-column interpretation, four Poseidon2 call composition, and
the before/after digest aliases. It consumes only the certified state-digest
slice and does not unfold the complete generated source artifact.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelationArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeStateDigestSliceCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Program

def beforeDigest
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Digest :=
  fun lane => ⟨assignment (2430 + lane.val), canonical _⟩

def afterDigest
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Digest :=
  fun lane => ⟨assignment (4603 + lane.val), canonical _⟩

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

private theorem zero_row_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {column : Nat}
    (holds : RowHolds assignment
      ⟨[(column, 1)], [(0, 1)], []⟩) :
    assignment column = 0 := by
  have columnLt := canonical column
  simp only [RowHolds, lcEval, List.foldl, one, goldilocksP] at holds columnLt
  omega

private theorem constant_row_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {column value : Nat}
    (valuePos : 0 < value)
    (valueLt : value < goldilocksP)
    (holds : RowHolds assignment
      ⟨[(0, goldilocksP - value), (column, 1)], [(0, 1)], []⟩) :
    assignment column = value := by
  have valueNe : value ≠ 0 := Nat.ne_of_gt valuePos
  have aPerm :
      [(0, goldilocksP - value), (column, 1)].Perm
        (builderLinearRow column [(0, value)]).a := by
    simpa [builderLinearRow, negateTerms, negCoeff, valueNe] using
      (List.perm_append_comm : List.Perm
        ([(0, goldilocksP - value)] ++ [(column, 1)])
        ([(column, 1)] ++ [(0, goldilocksP - value)]))
  have builderHolds :
      RowHolds assignment (builderLinearRow column [(0, value)]) :=
    rowHolds_of_operand_perms assignment
      (source :=
        ⟨[(0, goldilocksP - value), (column, 1)], [(0, 1)], []⟩)
      (target := builderLinearRow column [(0, value)])
      aPerm (List.Perm.refl _) (List.Perm.refl _) holds
  have defined := builderLinearRow_sound canonical one column [(0, value)]
    (by simp [CanonicalTerms, valuePos, valueLt]) builderHolds
  simpa [lcEval, one, Nat.mod_eq_of_lt valueLt] using defined

private theorem alias_row_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {source target : Nat}
    (holds : RowHolds assignment
      ⟨[(source, goldilocksP - 1), (target, 1)], [(0, 1)], []⟩) :
    assignment target = assignment source := by
  have aPerm :
      [(source, goldilocksP - 1), (target, 1)].Perm
        (builderLinearRow target [(source, 1)]).a := by
    simpa [builderLinearRow, negateTerms, negCoeff, goldilocksP] using
      (List.perm_append_comm : List.Perm
        ([(source, goldilocksP - 1)] ++ [(target, 1)])
        ([(target, 1)] ++ [(source, goldilocksP - 1)]))
  have builderHolds :
      RowHolds assignment (builderLinearRow target [(source, 1)]) :=
    rowHolds_of_operand_perms assignment
      (source :=
        ⟨[(source, goldilocksP - 1), (target, 1)], [(0, 1)], []⟩)
      (target := builderLinearRow target [(source, 1)])
      aPerm (List.Perm.refl _) (List.Perm.refl _) holds
  have defined := builderLinearRow_sound canonical one target [(source, 1)]
    (by simp [CanonicalTerms, goldilocksP]) builderHolds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical source)] using defined

private theorem prefix_row_holds
    {assignment : Nat → Nat}
    (satisfied : artifact.Satisfied assignment)
    {indexed : IndexedRow}
    (member : indexed ∈ prefixAndBeforeRows) :
    RowHolds assignment indexed.row :=
  satisfied.2.2 indexed (prefixAndBeforeRows_subset indexed member)

private theorem after_row_holds
    {assignment : Nat → Nat}
    (satisfied : artifact.Satisfied assignment)
    {indexed : IndexedRow}
    (member : indexed ∈ afterRows) :
    RowHolds assignment indexed.row :=
  satisfied.2.2 indexed (afterRows_subset indexed member)

private theorem initial_columns_zero
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (index : Fin 14) :
    assignment (1 + index.val) = 0 := by
  refine zero_row_sound canonical one ?_
  refine prefix_row_holds satisfied
    (indexed :=
      { index := index.val
        row := ⟨[(1 + index.val, 1)], [(0, 1)], []⟩ }) ?_
  rw [prefixAndBeforeRows_exact]
  fin_cases index <;> simp

private theorem collapsed_tail_exact
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment (19 + lane.val) =
      collapsedInitialValues.getD (4 + lane.val) 0 := by
  let value := collapsedInitialValues.getD (4 + lane.val) 0
  have valuePos : 0 < value := by
    fin_cases lane <;> native_decide
  have valueLt : value < goldilocksP := by
    fin_cases lane <;> native_decide
  change assignment (19 + lane.val) = value
  refine constant_row_sound canonical one
    (column := 19 + lane.val) (value := value) valuePos valueLt ?_
  refine prefix_row_holds satisfied
    (indexed :=
      { index := 18 + lane.val
        row := ⟨[(0, goldilocksP - value), (19 + lane.val, 1)],
          [(0, 1)], []⟩ }) ?_
  rw [prefixAndBeforeRows_exact]
  fin_cases lane <;> simp [value, collapsedInitialValues, goldilocksP]

private theorem frame_columns_exact
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment (23 + lane.val) = stateFieldsFrame10.getD lane.val 0 := by
  let value := stateFieldsFrame10.getD lane.val 0
  have valuePos : 0 < value := by
    fin_cases lane <;> native_decide
  have valueLt : value < goldilocksP := by
    fin_cases lane <;> native_decide
  change assignment (23 + lane.val) = value
  refine constant_row_sound canonical one
    (column := 23 + lane.val) (value := value) valuePos valueLt ?_
  refine prefix_row_holds satisfied
    (indexed :=
      { index := 22 + lane.val
        row := ⟨[(0, goldilocksP - value), (23 + lane.val, 1)],
          [(0, 1)], []⟩ }) ?_
  rw [prefixAndBeforeRows_exact]
  fin_cases lane <;> simp [value, stateFieldsFrame10, goldilocksP]

private theorem pad_column_exact
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    assignment 1827 = 1 := by
  refine constant_row_sound canonical one
    (column := 1827) (value := 1) (by native_decide) (by native_decide) ?_
  refine prefix_row_holds satisfied
    (indexed :=
      { index := 1826
        row := ⟨[(0, goldilocksP - 1), (1827, 1)], [(0, 1)], []⟩ }) ?_
  rw [prefixAndBeforeRows_exact]
  simp [goldilocksP]

private theorem before_alias_exact
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment (2430 + lane.val) = assignment (11 + lane.val) := by
  refine alias_row_sound canonical one ?_
  refine prefix_row_holds satisfied
    (indexed :=
      { index := 2429 + lane.val
        row := ⟨[(11 + lane.val, goldilocksP - 1),
          (2430 + lane.val, 1)], [(0, 1)], []⟩ }) ?_
  rw [prefixAndBeforeRows_exact]
  fin_cases lane <;> simp [goldilocksP]

private theorem after_alias_exact
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment (4603 + lane.val) = assignment (2420 + lane.val) := by
  refine alias_row_sound canonical one ?_
  refine after_row_holds satisfied
    (indexed :=
      { index := 4602 + lane.val
        row := ⟨[(2420 + lane.val, goldilocksP - 1),
          (4603 + lane.val, 1)], [(0, 1)], []⟩ }) ?_
  rw [afterRows_exact]
  fin_cases lane <;> simp [goldilocksP]

private theorem call_permutation_lane
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : Poseidon2Call.Call)
    (input : State)
    (callSatisfied : Satisfies call.rows assignment)
    (inputExact : ∀ lane : Fin 8,
      assignment (call.columnMap (lane.val + 1)) = (input.lanes lane).val)
    (lane : Fin 8) :
    assignment (call.columnMap (601 + lane.val)) =
      ((permute input).lanes lane).val := by
  have callSound := Poseidon2Call.rows_lanes_sound call canonical one
    callSatisfied lane.val (by omega)
  calc
    assignment (call.columnMap (601 + lane.val)) =
        Poseidon2PermutationSound.permute
          (fun inputLane => assignment (call.columnMap (inputLane + 1)))
          lane.val := callSound
    _ = Poseidon2PermutationSound.permute (laneNat input) lane.val := by
      apply Poseidon2PermutationSound.permute_congr
      intro inputLane inputLaneLt
      rw [inputExact ⟨inputLane, inputLaneLt⟩]
      simp [laneNat, width, inputLaneLt]
    _ = ((permute input).lanes lane).val := by
      have outputLt := Poseidon2PermutationSound.permute_lt
        (lanes := laneNat input) (by
          intro inputLane inputLaneLt
          simp [laneNat, width, inputLaneLt]) lane.val
      simp [permute, fieldValue, Nat.mod_eq_of_lt outputLt]

private theorem call3_output_column (lane : Fin 4) :
    call3.columnMap (601 + lane.val) = 2420 + lane.val := by
  fin_cases lane <;> rfl

/-- `PRELUDE-SOURCE-IMPLIES-HOLDS`: all exact compact source rows force the
independent native Prelude digest relation on the same assignment. -/
theorem source_rows_imply_holds
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    Holds productionSemantics
      (beforeDigest assignment canonical)
      (afterDigest assignment canonical) := by
  have call0Satisfied : Satisfies call0.rows assignment :=
    satisfied.2.1 call0 call0_member
  have call1Satisfied : Satisfies call1.rows assignment :=
    satisfied.2.1 call1 call1_member
  have call2Satisfied : Satisfies call2.rows assignment :=
    satisfied.2.1 call2 call2_member
  have call3Satisfied : Satisfies call3.rows assignment :=
    satisfied.2.1 call3 call3_member

  have call0Input : ∀ lane : Fin 8,
      assignment (call0.columnMap (lane.val + 1)) =
        (frameState.lanes lane).val := by
    intro lane
    fin_cases lane
    all_goals
      simp only [call0, Poseidon2Call.Call.columnMap, if_false, if_true,
        List.getD_cons_zero, Nat.reduceAdd]
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        frame_columns_exact canonical one satisfied ⟨0, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        frame_columns_exact canonical one satisfied ⟨1, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        frame_columns_exact canonical one satisfied ⟨2, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        frame_columns_exact canonical one satisfied ⟨3, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, collapsed_initial_state_state_exact,
        FPrimeFullHistoryStreamingPreludeDigestDomain.checkpointState,
        FPrimeFullHistoryStreamingPreludeDigestDomain.stateFromValues,
        fieldValue] using
        collapsed_tail_exact canonical one satisfied ⟨0, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, collapsed_initial_state_state_exact,
        FPrimeFullHistoryStreamingPreludeDigestDomain.checkpointState,
        FPrimeFullHistoryStreamingPreludeDigestDomain.stateFromValues,
        fieldValue] using
        collapsed_tail_exact canonical one satisfied ⟨1, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, collapsed_initial_state_state_exact,
        FPrimeFullHistoryStreamingPreludeDigestDomain.checkpointState,
        FPrimeFullHistoryStreamingPreludeDigestDomain.stateFromValues,
        fieldValue] using
        collapsed_tail_exact canonical one satisfied ⟨2, by decide⟩
    · simpa [frameState, stateFieldsFrame10, absorbWords, absorbElem,
        overwriteLane, collapsed_initial_state_state_exact,
        FPrimeFullHistoryStreamingPreludeDigestDomain.checkpointState,
        FPrimeFullHistoryStreamingPreludeDigestDomain.stateFromValues,
        fieldValue] using
        collapsed_tail_exact canonical one satisfied ⟨3, by decide⟩

  have call0Output : ∀ lane : Fin 8,
      assignment (call0.columnMap (601 + lane.val)) =
        ((permute frameState).lanes lane).val :=
    call_permutation_lane canonical one call0 frameState call0Satisfied
      call0Input

  have call1Input : ∀ lane : Fin 8,
      assignment (call1.columnMap (lane.val + 1)) =
        (firstFieldsState.lanes lane).val := by
    intro lane
    fin_cases lane
    all_goals
      simp only [call1, Poseidon2Call.Call.columnMap, if_false, if_true,
        List.getD_cons_zero, Nat.reduceAdd]
    · simpa [firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨0, by decide⟩
    · simpa [firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨1, by decide⟩
    · simpa [firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨2, by decide⟩
    · simpa [firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨3, by decide⟩
    · simpa [call0, firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call0Output ⟨4, by decide⟩
    · simpa [call0, firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call0Output ⟨5, by decide⟩
    · simpa [call0, firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call0Output ⟨6, by decide⟩
    · simpa [call0, firstFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call0Output ⟨7, by decide⟩

  have call1Output : ∀ lane : Fin 8,
      assignment (call1.columnMap (601 + lane.val)) =
        ((permute firstFieldsState).lanes lane).val :=
    call_permutation_lane canonical one call1 firstFieldsState call1Satisfied
      call1Input

  have call2Input : ∀ lane : Fin 8,
      assignment (call2.columnMap (lane.val + 1)) =
        (secondFieldsState.lanes lane).val := by
    intro lane
    fin_cases lane
    all_goals
      simp only [call2, Poseidon2Call.Call.columnMap, if_false, if_true,
        List.getD_cons_zero, Nat.reduceAdd]
    · simpa [secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨4, by decide⟩
    · simpa [secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨5, by decide⟩
    · simpa [secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨6, by decide⟩
    · simpa [secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨7, by decide⟩
    · simpa [call1, secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call1Output ⟨4, by decide⟩
    · simpa [call1, secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call1Output ⟨5, by decide⟩
    · simpa [call1, secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call1Output ⟨6, by decide⟩
    · simpa [call1, secondFieldsState, zeroBlock4, absorbWords, absorbElem,
        overwriteLane] using call1Output ⟨7, by decide⟩

  have call2Output : ∀ lane : Fin 8,
      assignment (call2.columnMap (601 + lane.val)) =
        ((permute secondFieldsState).lanes lane).val :=
    call_permutation_lane canonical one call2 secondFieldsState call2Satisfied
      call2Input

  have call3Input : ∀ lane : Fin 8,
      assignment (call3.columnMap (lane.val + 1)) =
        (digestInputState.lanes lane).val := by
    intro lane
    fin_cases lane
    all_goals
      simp only [call3, Poseidon2Call.Call.columnMap, if_false, if_true,
        List.getD_cons_zero, Nat.reduceAdd]
    · simpa [digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨8, by decide⟩
    · simpa [digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        initial_columns_zero canonical one satisfied ⟨9, by decide⟩
    · simpa [digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane, wordField, fieldValue] using
        pad_column_exact canonical one satisfied
    · simpa [call2, digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane] using call2Output ⟨3, by decide⟩
    · simpa [call2, digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane] using call2Output ⟨4, by decide⟩
    · simpa [call2, digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane] using call2Output ⟨5, by decide⟩
    · simpa [call2, digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane] using call2Output ⟨6, by decide⟩
    · simpa [call2, digestInputState, zeroBlock2, absorbWords, absorbElem,
        overwriteLane] using call2Output ⟨7, by decide⟩

  have call3Output : ∀ lane : Fin 8,
      assignment (call3.columnMap (601 + lane.val)) =
        (scheduledInitialDigestState.lanes lane).val := by
    intro lane
    simpa only [scheduledInitialDigestState] using
      call_permutation_lane canonical one call3 digestInputState
        call3Satisfied call3Input lane

  constructor
  · funext lane
    apply Fin.ext
    change assignment (2430 + lane.val) = 0
    rw [before_alias_exact canonical one satisfied lane]
    have zero := initial_columns_zero canonical one satisfied
      ⟨10 + lane.val, by omega⟩
    have columnEq : 1 + (10 + lane.val) = 11 + lane.val := by omega
    rw [columnEq] at zero
    exact zero
  · funext lane
    apply Fin.ext
    change assignment (4603 + lane.val) =
      (stateDigest initialReplayFields lane).val
    rw [after_alias_exact canonical one satisfied lane,
      stateDigest_initial_exact]
    have output := call3Output ⟨lane.val, by omega⟩
    rw [call3_output_column lane] at output
    simpa using output

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelationArtifact
