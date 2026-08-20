import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema

/-!
Contract: reusable structural validity certificate for a variable-length
Poseidon2 hash recipe.

Owns round validity, round linkage, complete input absorption, the four-lane
final output, and the terminal pad round. Concrete artifacts must separately
prove their input coverage and output-column identity.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

private theorem callOutputColumns_exact
    (recipe : VariableHashRecipe) (index : Nat) :
    recipe.callOutputColumns index =
      eight.map fun lane => (recipe.call index).columnMap (601 + lane) := by
  rw [VariableHashRecipe.callOutputColumns, List.range'_eq_map_range]
  apply List.map_congr_left
  intro lane member
  have laneLt : lane < 8 := List.mem_range.mp member
  have nonzero : 601 + lane ≠ 0 := by omega
  have notSmall : ¬ 601 + lane < 9 := by omega
  simp [VariableHashRecipe.call, Call.columnMap, nonzero, notSmall]
  omega

private theorem stateBeforeColumns_length
    (recipe : VariableHashRecipe) (index : Nat) :
    (recipe.stateBeforeColumns index).length = 8 := by
  unfold VariableHashRecipe.stateBeforeColumns VariableHashRecipe.callOutputColumns
  split <;> simp

private theorem chunkColumns_length_le_four
    (recipe : VariableHashRecipe) (index : Nat) :
    (recipe.chunkColumns index).length ≤ 4 := by
  simp [VariableHashRecipe.chunkColumns, List.length_take, rate]

private theorem absorbRound_metadataValid
    (recipe : VariableHashRecipe) {index : Nat}
    (indexLt : index < recipe.absorbRounds) :
    (recipe.absorbRound index).metadataValid := by
  have stateLength := stateBeforeColumns_length recipe index
  have chunkBound := chunkColumns_length_le_four recipe index
  refine ⟨stateLength, ?_, by simp [VariableHashRecipe.absorbRound,
      VariableHashRecipe.callOutputColumns], rfl,
    callOutputColumns_exact recipe index, ?_⟩
  · simp [VariableHashRecipe.absorbRound, VariableHashRecipe.callInputColumns,
      VariableHashRecipe.definitionCount, indexLt, stateLength, chunkBound] <;>
      omega
  · simp [VariableHashRecipe.absorbRound, VariableHashRecipe.callInputColumns,
      VariableHashRecipe.definitionCount, indexLt, stateLength, chunkBound]

private theorem call_rows_length
    (recipe : VariableHashRecipe) (index : Nat) :
    (recipe.call index).rows.length = 600 := by
  rw [Call.rows, List.length_map, Poseidon2Permutation.rows_length]
  rfl

private theorem absorbRound_valid
    (recipe : VariableHashRecipe) {index : Nat}
    (indexLt : index < recipe.absorbRounds) :
    (recipe.absorbRound index).Valid (recipe.absorbRound index).rows := by
  apply Round.selfValid (absorbRound_metadataValid recipe indexLt)
  · simp [VariableHashRecipe.absorbRound, VariableHashRecipe.call,
      VariableHashRecipe.definitionCount, indexLt,
      Round.expectedDefinitionRows]
  · simpa [VariableHashRecipe.absorbRound, VariableHashRecipe.call,
      VariableHashRecipe.definitionCount, indexLt,
      Round.expectedDefinitionRows] using
      (call_rows_length recipe index).symm

private theorem padRound_metadataValid (recipe : VariableHashRecipe) :
    recipe.padRound.metadataValid := by
  refine ⟨stateBeforeColumns_length recipe recipe.absorbRounds, ?_,
    by simp [VariableHashRecipe.padRound,
      VariableHashRecipe.callOutputColumns], rfl,
    callOutputColumns_exact recipe recipe.absorbRounds, ?_⟩
  · simp [VariableHashRecipe.padRound, VariableHashRecipe.callInputColumns,
      stateBeforeColumns_length]
  · simp [VariableHashRecipe.padRound, VariableHashRecipe.callInputColumns,
      stateBeforeColumns_length]

private theorem padRound_valid (recipe : VariableHashRecipe) :
    recipe.padRound.Valid recipe.padRound.rows := by
  apply Round.selfValid (padRound_metadataValid recipe)
  · simp [VariableHashRecipe.padRound, VariableHashRecipe.call,
      VariableHashRecipe.definitionCount, Round.expectedDefinitionRows]
  · simpa [VariableHashRecipe.padRound, VariableHashRecipe.call,
      VariableHashRecipe.definitionCount, Round.expectedDefinitionRows] using
      (call_rows_length recipe recipe.absorbRounds).symm

private theorem linkedCheck_append
    (prior : List Nat) (left right : List Round) :
    linkedCheck prior (left ++ right) =
      (linkedCheck prior left &&
        linkedCheck (finalColumns prior left) right) := by
  induction left generalizing prior with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp [linkedCheck, finalColumns, inductionHypothesis, Bool.and_assoc]

private theorem linkedCheck_absorbRounds
    (recipe : VariableHashRecipe) (start count : Nat) :
    linkedCheck (recipe.stateBeforeColumns start)
      ((List.range' start count).map recipe.absorbRound) = true := by
  induction count generalizing start with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range'_succ]
      simp only [List.map_cons, linkedCheck, VariableHashRecipe.absorbRound,
        decide_true, Bool.true_and]
      have nextState :
          recipe.stateBeforeColumns (start + 1) =
            recipe.callOutputColumns start := by
        simp [VariableHashRecipe.stateBeforeColumns]
      rw [← nextState]
      exact inductionHypothesis (start + 1)

private theorem finalColumns_append_singleton
    (prior : List Nat) (rounds : List Round) (last : Round) :
    finalColumns prior (rounds ++ [last]) =
      last.permutationOutputColumns := by
  induction rounds generalizing prior with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [List.cons_append, finalColumns]
      exact inductionHypothesis round.permutationOutputColumns

private theorem absorbed_chunk_prefix
    (values : List Nat) (count : Nat) :
    (List.range count).flatMap
        (fun index => (values.drop (4 * index)).take 4) =
      values.take (4 * count) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_singleton, inductionHypothesis]
      rw [Nat.mul_succ, List.take_add]

private theorem trace_roundsAccepted (recipe : VariableHashRecipe) :
    recipe.trace.rounds.all
        (fun round => decide (round.Valid round.rows)) = true := by
  apply List.all_eq_true.mpr
  intro round member
  rw [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    List.mem_append] at member
  rcases member with member | member
  · rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    exact decide_eq_true
      (absorbRound_valid recipe (List.mem_range.mp indexMember))
  · simp only [List.mem_singleton] at member
    subst round
    exact decide_eq_true (padRound_valid recipe)

private theorem trace_linked
    (recipe : VariableHashRecipe) (positive : 0 < recipe.absorbRounds) :
    linkedCheck (List.replicate 8 recipe.trace.zeroColumn)
        recipe.trace.rounds = true := by
  rw [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    List.range_eq_range', linkedCheck_append]
  have absorbs :=
    linkedCheck_absorbRounds recipe 0 recipe.absorbRounds
  have initial :
      recipe.stateBeforeColumns 0 =
        List.replicate 8 recipe.zeroColumn := by
    rfl
  rw [← initial, absorbs, Bool.true_and]
  have finalAbsorb :
      finalColumns (recipe.stateBeforeColumns 0)
          ((List.range' 0 recipe.absorbRounds).map
            recipe.absorbRound) =
        recipe.callOutputColumns (recipe.absorbRounds - 1) := by
    have count :
        recipe.absorbRounds = (recipe.absorbRounds - 1) + 1 := by
      omega
    conv_lhs =>
      rw [count, List.range'_concat, List.map_append]
    simpa [VariableHashRecipe.absorbRound] using
      finalColumns_append_singleton (recipe.stateBeforeColumns 0)
        ((List.range' 0 (recipe.absorbRounds - 1)).map
          recipe.absorbRound)
        (recipe.absorbRound (recipe.absorbRounds - 1))
  rw [finalAbsorb]
  simp only [linkedCheck, Bool.and_eq_true]
  constructor
  · apply decide_eq_true
    simp [VariableHashRecipe.padRound,
      VariableHashRecipe.stateBeforeColumns]
    omega
  · trivial

private theorem trace_inputsOwned
    (recipe : VariableHashRecipe)
    (coverage :
      recipe.inputColumns.length ≤ 4 * recipe.absorbRounds) :
    recipe.trace.absorbedColumns = recipe.trace.inputColumns := by
  rw [VariableHashRecipe.trace, Trace.absorbedColumns,
    VariableHashRecipe.rounds]
  simp only [absorbedColumnsOf, List.flatMap_append,
    List.flatMap_map, VariableHashRecipe.absorbRound,
    VariableHashRecipe.padRound, List.flatMap_singleton,
    List.append_nil]
  calc
    (List.range recipe.absorbRounds).flatMap recipe.chunkColumns =
        recipe.inputColumns.take (4 * recipe.absorbRounds) := by
      simpa [VariableHashRecipe.chunkColumns] using
        absorbed_chunk_prefix recipe.inputColumns recipe.absorbRounds
    _ = recipe.inputColumns := by
      apply List.take_of_length_le
      exact coverage

private theorem trace_finalOutput
    (recipe : VariableHashRecipe)
    (outputExact :
      recipe.outputColumns =
        (recipe.callOutputColumns recipe.absorbRounds).take 4) :
    recipe.trace.outputColumns =
      (finalColumns (List.replicate 8 recipe.trace.zeroColumn)
        recipe.trace.rounds).take 4 := by
  rw [VariableHashRecipe.trace, VariableHashRecipe.rounds]
  rw [finalColumns_append_singleton]
  simpa [VariableHashRecipe.padRound] using outputExact

private theorem trace_terminalPad (recipe : VariableHashRecipe) :
    recipe.trace.rounds.getLast?.map Round.kind = some .pad := by
  simp [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    VariableHashRecipe.padRound]

private theorem chunkColumns_length_eq_rate
    (recipe : VariableHashRecipe) {index : Nat}
    (indexLt : index < recipe.absorbRounds)
    (full :
      recipe.inputColumns.length = rate * recipe.absorbRounds) :
    (recipe.chunkColumns index).length = rate := by
  simp only [VariableHashRecipe.chunkColumns, List.length_take,
    List.length_drop]
  rw [full]
  unfold rate
  omega

private theorem definitionCount_eq_rate
    (recipe : VariableHashRecipe) {index : Nat}
    (indexLt : index < recipe.absorbRounds)
    (full :
      recipe.inputColumns.length = rate * recipe.absorbRounds) :
    recipe.definitionCount index = rate := by
  simp [VariableHashRecipe.definitionCount, indexLt,
    chunkColumns_length_eq_rate recipe indexLt full]

private theorem allocatedBefore_eq_of_fullAbsorbRounds
    (recipe : VariableHashRecipe)
    (full :
      recipe.inputColumns.length = rate * recipe.absorbRounds)
    {count : Nat} (countLe : count ≤ recipe.absorbRounds) :
    recipe.allocatedBefore count =
      count * (rate + permutationRows) := by
  induction count with
  | zero => simp [VariableHashRecipe.allocatedBefore]
  | succ count inductionHypothesis =>
      have countLt : count < recipe.absorbRounds := by omega
      rw [VariableHashRecipe.allocatedBefore, List.range_succ,
        List.map_append, List.sum_append]
      simp only [List.map_singleton, List.sum_singleton]
      change recipe.allocatedBefore count +
        (recipe.definitionCount count + permutationRows) = _
      have prior :
          recipe.allocatedBefore count =
            count * (rate + permutationRows) :=
        inductionHypothesis (by omega)
      have current : recipe.definitionCount count = rate :=
        definitionCount_eq_rate recipe countLt full
      calc
        _ = count * (rate + permutationRows) +
            (recipe.definitionCount count + permutationRows) := by
          rw [prior]
        _ = count * (rate + permutationRows) +
            (rate + permutationRows) := by
          rw [current]
        _ = (count + 1) * (rate + permutationRows) := by
          simpa only [Nat.succ_eq_add_one] using
            (Nat.succ_mul count (rate + permutationRows)).symm

/-- For a whole number of rate-sized chunks, compute the final pad-call
output columns without reducing the complete round list. -/
theorem finalCallOutputColumns_eq_of_fullAbsorbRounds
    (recipe : VariableHashRecipe)
    (full :
      recipe.inputColumns.length = rate * recipe.absorbRounds) :
    recipe.callOutputColumns recipe.absorbRounds =
      List.range'
        (recipe.zeroColumn + 1 +
          recipe.absorbRounds * (rate + permutationRows) + 1 + 592) 8 := by
  rw [VariableHashRecipe.callOutputColumns,
    VariableHashRecipe.callFirstAllocatedColumn,
    VariableHashRecipe.roundColumnStart,
    allocatedBefore_eq_of_fullAbsorbRounds recipe full (Nat.le_refl _)]
  simp [VariableHashRecipe.definitionCount]

/-- One variable recipe is structurally complete when its absorb rounds cover
the full input and its declared outputs are the final four sponge lanes. -/
theorem ownedValid
    (recipe : VariableHashRecipe)
    (positive : 0 < recipe.absorbRounds)
    (coverage :
      recipe.inputColumns.length ≤ 4 * recipe.absorbRounds)
    (outputExact :
      recipe.outputColumns =
        (recipe.callOutputColumns recipe.absorbRounds).take 4) :
    recipe.trace.OwnedValid := by
  exact {
    roundsAccepted := trace_roundsAccepted recipe
    linked := trace_linked recipe positive
    inputsOwned := trace_inputsOwned recipe coverage
    finalOutput := trace_finalOutput recipe outputExact
    outputLength := by
      change recipe.outputColumns.length = 4
      rw [outputExact]
      simp [VariableHashRecipe.callOutputColumns]
    terminalPad := trace_terminalPad recipe
  }

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
