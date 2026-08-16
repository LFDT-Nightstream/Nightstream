import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope

/-!
Contract: structural validity certificate for the Rust-emitted PiRLC phase
envelope artifact.

Assurance tier: Rust-to-Lean artifact geometry certificate.

Owns the formula-level sponge schedule proof and the small exact arm-layout
facts. It does not evaluate the expanded 662,971-row phase programs.

Does not own phase semantics, lifecycle links, selective lowering, or
collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

private theorem callOutputColumns_exact
    (recipe : HashRecipe) (index : Nat) :
    recipe.callOutputColumns index =
      eight.map fun lane => (recipe.call index).columnMap (601 + lane) := by
  rw [HashRecipe.callOutputColumns, List.range'_eq_map_range]
  apply List.map_congr_left
  intro lane member
  have laneLt : lane < 8 := List.mem_range.mp member
  have nonzero : 601 + lane ≠ 0 := by omega
  have notSmall : ¬ 601 + lane < 9 := by omega
  simp [HashRecipe.call, Call.columnMap, nonzero, notSmall]
  omega

private theorem stateBeforeColumns_length
    (recipe : HashRecipe) (index : Nat) :
    (recipe.stateBeforeColumns index).length = 8 := by
  unfold HashRecipe.stateBeforeColumns HashRecipe.callOutputColumns
  split <;> simp

private theorem chunkColumns_length
    (recipe : HashRecipe) (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    (recipe.chunkColumns index).length = 4 := by
  have enough : 4 ≤ hashInputFields - 4 * index := by
    norm_num [hashInputFields, hashConstantFields, domainFields, digestFields,
      payloadFields, absorbRounds] at indexLt ⊢
    omega
  simp [HashRecipe.chunkColumns, List.length_take, List.length_drop,
    inputLength, Nat.min_eq_left enough]

private theorem absorbRound_metadataValid
    (recipe : HashRecipe) (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    (recipe.absorbRound index).metadataValid := by
  have stateLength := stateBeforeColumns_length recipe index
  have chunkLength := chunkColumns_length recipe inputLength indexLt
  refine ⟨stateLength, ?_, by simp [HashRecipe.absorbRound,
      HashRecipe.callOutputColumns], rfl, callOutputColumns_exact recipe index, ?_⟩
  · simp [HashRecipe.absorbRound, HashRecipe.callInputColumns, indexLt,
      stateLength]
  · simp [HashRecipe.absorbRound, HashRecipe.callInputColumns, indexLt,
      chunkLength, stateLength]

private theorem call_rows_length (recipe : HashRecipe) (index : Nat) :
    (recipe.call index).rows.length = permutationRows := by
  rw [Call.rows, List.length_map, Poseidon2Permutation.rows_length]
  rfl

private theorem absorbRound_valid
    (recipe : HashRecipe) (inputLength : recipe.inputColumns.length = hashInputFields)
    {index : Nat} (indexLt : index < absorbRounds) :
    (recipe.absorbRound index).Valid (recipe.absorbRound index).rows := by
  apply Round.selfValid (absorbRound_metadataValid recipe inputLength indexLt)
  · simp [HashRecipe.absorbRound, HashRecipe.call, HashRecipe.definitionCount,
      indexLt, Round.expectedDefinitionRows,
      chunkColumns_length recipe inputLength indexLt]
  · simpa [HashRecipe.absorbRound, HashRecipe.call,
      HashRecipe.definitionCount, indexLt, Round.expectedDefinitionRows,
      chunkColumns_length recipe inputLength indexLt] using
      (call_rows_length recipe index).symm

private theorem padRound_metadataValid (recipe : HashRecipe) :
    recipe.padRound.metadataValid := by
  refine ⟨stateBeforeColumns_length recipe absorbRounds, ?_,
    by simp [HashRecipe.padRound,
      HashRecipe.callOutputColumns], rfl,
    callOutputColumns_exact recipe absorbRounds, ?_⟩
  · simp [HashRecipe.padRound, HashRecipe.callInputColumns, absorbRounds,
      stateBeforeColumns_length]
  · simp [HashRecipe.padRound, HashRecipe.callInputColumns, absorbRounds,
      stateBeforeColumns_length]

private theorem padRound_valid (recipe : HashRecipe) :
    recipe.padRound.Valid recipe.padRound.rows := by
  apply Round.selfValid (padRound_metadataValid recipe)
  · simp [HashRecipe.padRound, HashRecipe.call, HashRecipe.definitionCount,
      absorbRounds, Round.expectedDefinitionRows]
  · simpa [HashRecipe.padRound, HashRecipe.call,
      HashRecipe.definitionCount, absorbRounds,
      Round.expectedDefinitionRows] using
      (call_rows_length recipe absorbRounds).symm

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
    (recipe : HashRecipe) (start count : Nat) :
    linkedCheck (recipe.stateBeforeColumns start)
      ((List.range' start count).map recipe.absorbRound) = true := by
  induction count generalizing start with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range'_succ]
      simp only [List.map_cons, linkedCheck, HashRecipe.absorbRound,
        decide_true, Bool.true_and]
      have nextState :
          recipe.stateBeforeColumns (start + 1) =
            recipe.callOutputColumns start := by
        simp [HashRecipe.stateBeforeColumns]
      rw [← nextState]
      exact inductionHypothesis (start + 1)

private theorem finalColumns_append_singleton
    (prior : List Nat) (rounds : List Round) (last : Round) :
    finalColumns prior (rounds ++ [last]) = last.permutationOutputColumns := by
  induction rounds generalizing prior with
  | nil => rfl
  | cons round rest inductionHypothesis =>
      simp only [List.cons_append, finalColumns]
      exact inductionHypothesis round.permutationOutputColumns

private theorem absorbed_chunk_prefix (values : List Nat) (count : Nat) :
    (List.range count).flatMap
        (fun index => (values.drop (4 * index)).take 4) =
      values.take (4 * count) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_singleton, inductionHypothesis]
      rw [Nat.mul_succ, List.take_add]

private theorem trace_roundsAccepted
    (recipe : HashRecipe) (inputLength : recipe.inputColumns.length = hashInputFields) :
    recipe.trace.rounds.all
        (fun round => decide (round.Valid round.rows)) = true := by
  apply List.all_eq_true.mpr
  intro round member
  rw [HashRecipe.trace, HashRecipe.rounds, List.mem_append] at member
  rcases member with member | member
  · rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    exact decide_eq_true
      (absorbRound_valid recipe inputLength (List.mem_range.mp indexMember))
  · simp only [List.mem_singleton] at member
    subst round
    exact decide_eq_true (padRound_valid recipe)

private theorem trace_linked (recipe : HashRecipe) :
    linkedCheck (List.replicate 8 recipe.trace.zeroColumn)
        recipe.trace.rounds = true := by
  rw [HashRecipe.trace, HashRecipe.rounds, List.range_eq_range',
    linkedCheck_append]
  have absorbs := linkedCheck_absorbRounds recipe 0 absorbRounds
  have initial :
      recipe.stateBeforeColumns 0 =
        List.replicate 8 recipe.zeroColumn := by
    rfl
  rw [← initial, absorbs, Bool.true_and]
  have finalAbsorb :
      finalColumns (recipe.stateBeforeColumns 0)
          ((List.range' 0 absorbRounds).map recipe.absorbRound) =
        recipe.callOutputColumns (absorbRounds - 1) := by
    have count : absorbRounds = (absorbRounds - 1) + 1 := by
      norm_num [absorbRounds, hashInputFields, hashConstantFields,
        domainFields, digestFields, payloadFields]
    conv_lhs =>
      rw [count, List.range'_concat, List.map_append]
    simpa [HashRecipe.absorbRound] using
      finalColumns_append_singleton (recipe.stateBeforeColumns 0)
        ((List.range' 0 (absorbRounds - 1)).map recipe.absorbRound)
        (recipe.absorbRound (absorbRounds - 1))
  rw [finalAbsorb]
  simp only [linkedCheck, Bool.and_eq_true]
  constructor
  · apply decide_eq_true
    simp [HashRecipe.padRound, HashRecipe.stateBeforeColumns]
    norm_num [absorbRounds, hashInputFields, hashConstantFields,
      domainFields, digestFields, payloadFields]
  · trivial

private theorem trace_inputsOwned
    (recipe : HashRecipe) (inputLength : recipe.inputColumns.length = hashInputFields) :
    recipe.trace.absorbedColumns = recipe.trace.inputColumns := by
  rw [HashRecipe.trace, Trace.absorbedColumns, HashRecipe.rounds]
  simp only [absorbedColumnsOf, List.flatMap_append, List.flatMap_map,
    HashRecipe.absorbRound, HashRecipe.padRound, List.flatMap_singleton,
    List.append_nil]
  have exactFields : 4 * absorbRounds = hashInputFields := by
    norm_num [absorbRounds, hashInputFields, hashConstantFields, domainFields,
      digestFields, payloadFields]
  calc
    (List.range absorbRounds).flatMap recipe.chunkColumns =
        recipe.inputColumns.take (4 * absorbRounds) := by
      simpa [HashRecipe.chunkColumns] using
        absorbed_chunk_prefix recipe.inputColumns absorbRounds
    _ = recipe.inputColumns := by
      rw [exactFields]
      apply List.take_of_length_le
      omega

private theorem trace_finalOutput
    (recipe : HashRecipe)
    (outputExact :
      recipe.outputColumns = (recipe.callOutputColumns absorbRounds).take 4) :
    recipe.trace.outputColumns =
      (finalColumns (List.replicate 8 recipe.trace.zeroColumn)
        recipe.trace.rounds).take 4 := by
  rw [HashRecipe.trace, HashRecipe.rounds]
  rw [finalColumns_append_singleton]
  simpa [HashRecipe.padRound] using outputExact

private theorem trace_terminalPad (recipe : HashRecipe) :
    recipe.trace.rounds.getLast?.map Round.kind = some .pad := by
  simp [HashRecipe.trace, HashRecipe.rounds, HashRecipe.padRound]

theorem hashRecipe_trace_ownedValid
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields)
    (outputExact :
      recipe.outputColumns = (recipe.callOutputColumns absorbRounds).take 4) :
    recipe.trace.OwnedValid := by
  exact {
    roundsAccepted := trace_roundsAccepted recipe inputLength
    linked := trace_linked recipe
    inputsOwned := trace_inputsOwned recipe inputLength
    finalOutput := trace_finalOutput recipe outputExact
    outputLength := by
      change recipe.outputColumns.length = 4
      rw [outputExact]
      simp [HashRecipe.callOutputColumns]
    terminalPad := trace_terminalPad recipe
  }

theorem hashRecipe_valueSchedules_exact
    (recipe : HashRecipe)
    (inputLength : recipe.inputColumns.length = hashInputFields) :
    valueSchedules recipe.trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] := by
  rw [HashRecipe.trace, HashRecipe.rounds, valueSchedules, List.map_append]
  simp only [List.map_singleton, HashRecipe.padRound, Round.valueSchedule]
  congr 1
  rw [List.map_map]
  calc
    (List.range absorbRounds).map
        (fun index => (recipe.absorbRound index).valueSchedule) =
      (List.range absorbRounds).map (fun _index => .absorb 4) := by
        apply List.map_congr_left
        intro index member
        simp only [HashRecipe.absorbRound, Round.valueSchedule]
        rw [chunkColumns_length recipe inputLength
          (List.mem_range.mp member)]
    _ = List.replicate absorbRounds (.absorb 4) := by simp

private theorem columnsValid_range'
    (columnCount start length : Nat)
    (bounded : start + length ≤ columnCount) :
    columnsValid columnCount length (List.range' start length) := by
  refine ⟨by simp, List.nodup_range', ?_⟩
  intro column member
  rw [List.mem_range'_1] at member
  omega

private theorem hashRecipe_valid
    (recipe : HashRecipe) (columnCount : Nat)
    (constantLength : recipe.constantValues.length = hashConstantFields)
    (constantCanonical :
      ∀ value ∈ recipe.constantValues, 0 < value ∧ value < goldilocksP)
    (constantColumns :
      columnsValid columnCount hashConstantFields recipe.constantColumns)
    (localColumns :
      columnsValid columnCount digestFields recipe.localColumns)
    (payloadColumns :
      columnsValid columnCount payloadFields recipe.payloadColumns)
    (outputColumns :
      columnsValid columnCount digestFields recipe.outputColumns)
    (outputExact :
      recipe.outputColumns = (recipe.callOutputColumns absorbRounds).take 4) :
    recipe.Valid columnCount := by
  have inputLength : recipe.inputColumns.length = hashInputFields := by
    simp [HashRecipe.inputColumns, HashRecipe.constantColumns,
      constantLength, constantColumns.1, localColumns.1, payloadColumns.1,
      hashInputFields, hashConstantFields]
    omega
  exact ⟨constantLength, constantCanonical, constantColumns, localColumns,
    payloadColumns, outputColumns, inputLength,
    hashRecipe_trace_ownedValid recipe inputLength outputExact⟩

private theorem phaseConstantValues_canonical :
    ∀ value ∈ phaseConstantValues, 0 < value ∧ value < goldilocksP := by
  norm_num [phaseConstantValues, goldilocksP]

theorem evenBeforeHash_valid :
    (evenArm.hashRecipe phaseConstantValues .before).Valid
      evenArm.bodyColumns := by
  apply hashRecipe_valid
  · rfl
  · exact phaseConstantValues_canonical
  · change columnsValid 1233086 11 (List.range' 560785 11)
    exact columnsValid_range' 1233086 560785 11 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.localAliasColumns, evenArm]
  · change columnsValid 1233086 2169 (List.range' 558612 2169)
    exact columnsValid_range' 1233086 558612 2169 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.semanticDigestColumns, evenArm]
  · rfl

theorem evenAfterHash_valid :
    (evenArm.hashRecipe phaseConstantValues .after).Valid
      evenArm.bodyColumns := by
  apply hashRecipe_valid
  · rfl
  · exact phaseConstantValues_canonical
  · change columnsValid 1233086 11 (List.range' 891182 11)
    exact columnsValid_range' 1233086 891182 11 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.localAliasColumns, evenArm]
  · change columnsValid 1233086 2169 (List.range' 558612 2169)
    exact columnsValid_range' 1233086 558612 2169 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.semanticDigestColumns, evenArm]
  · rfl

theorem oddBeforeHash_valid :
    (oddArm.hashRecipe phaseConstantValues .before).Valid
      oddArm.bodyColumns := by
  apply hashRecipe_valid
  · rfl
  · exact phaseConstantValues_canonical
  · change columnsValid 1234286 11 (List.range' 561985 11)
    exact columnsValid_range' 1234286 561985 11 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.localAliasColumns, oddArm]
  · change columnsValid 1234286 2169 (List.range' 559812 2169)
    exact columnsValid_range' 1234286 559812 2169 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.semanticDigestColumns, oddArm]
  · rfl

theorem oddAfterHash_valid :
    (oddArm.hashRecipe phaseConstantValues .after).Valid
      oddArm.bodyColumns := by
  apply hashRecipe_valid
  · rfl
  · exact phaseConstantValues_canonical
  · change columnsValid 1234286 11 (List.range' 892382 11)
    exact columnsValid_range' 1234286 892382 11 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.localAliasColumns, oddArm]
  · change columnsValid 1234286 2169 (List.range' 559812 2169)
    exact columnsValid_range' 1234286 559812 2169 (by omega)
  · norm_num [columnsValid, digestFields, RawArm.hashRecipe,
      RawArm.semanticDigestColumns, oddArm]
  · rfl

theorem evenBeforeHash_trace_ownedValid :
    (evenArm.hashRecipe phaseConstantValues .before).trace.OwnedValid :=
  evenBeforeHash_valid.2.2.2.2.2.2.2

theorem evenAfterHash_trace_ownedValid :
    (evenArm.hashRecipe phaseConstantValues .after).trace.OwnedValid :=
  evenAfterHash_valid.2.2.2.2.2.2.2

theorem oddBeforeHash_trace_ownedValid :
    (oddArm.hashRecipe phaseConstantValues .before).trace.OwnedValid :=
  oddBeforeHash_valid.2.2.2.2.2.2.2

theorem oddAfterHash_trace_ownedValid :
    (oddArm.hashRecipe phaseConstantValues .after).trace.OwnedValid :=
  oddAfterHash_valid.2.2.2.2.2.2.2

theorem evenBeforeHash_valueSchedules_exact :
    valueSchedules
        (evenArm.hashRecipe phaseConstantValues .before).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  hashRecipe_valueSchedules_exact _ evenBeforeHash_valid.2.2.2.2.2.2.1

theorem evenAfterHash_valueSchedules_exact :
    valueSchedules
        (evenArm.hashRecipe phaseConstantValues .after).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  hashRecipe_valueSchedules_exact _ evenAfterHash_valid.2.2.2.2.2.2.1

theorem oddBeforeHash_valueSchedules_exact :
    valueSchedules
        (oddArm.hashRecipe phaseConstantValues .before).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  hashRecipe_valueSchedules_exact _ oddBeforeHash_valid.2.2.2.2.2.2.1

theorem oddAfterHash_valueSchedules_exact :
    valueSchedules
        (oddArm.hashRecipe phaseConstantValues .after).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  hashRecipe_valueSchedules_exact _ oddAfterHash_valid.2.2.2.2.2.2.1

theorem evenArm_valid : evenArm.Valid phaseConstantValues := by
  refine ⟨by decide, by decide, rfl, rfl, by norm_num [evenArm],
    by norm_num [evenArm], ?_, ?_, ?_, ?_, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, evenBeforeHash_valid, evenAfterHash_valid⟩
  · norm_num [columnsValid, digestFields, evenArm]
  · norm_num [columnsValid, digestFields, evenArm]
  · norm_num [evenArm]
  · norm_num [evenArm]

theorem oddArm_valid : oddArm.Valid phaseConstantValues := by
  refine ⟨by decide, by decide, rfl, rfl, by norm_num [oddArm],
    by norm_num [oddArm], ?_, ?_, ?_, ?_, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, oddBeforeHash_valid, oddAfterHash_valid⟩
  · norm_num [columnsValid, digestFields, oddArm]
  · norm_num [columnsValid, digestFields, oddArm]
  · norm_num [oddArm]
  · norm_num [oddArm]

theorem rawArtifact_valid : rawArtifact.Valid := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl, evenArm_valid, oddArm_valid⟩

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope
