import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge

/-!
Contract: exact nonzero-coefficient accounting for the selected fixed-23
Poseidon2 sponge.

Owns: survival under a fresh round-constant wire, the bounded seven-call first
round certificate, exact scheduled-input lengths, and the receipt-folded
program total.

Does not own: the Rust coefficient count or typed hash-call activation/output
rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

theorem selected_constants_nonzero :
    (∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      selected.initial round.val lane % goldilocksP ≠ 0) ∧
    (∀ round : Fin partialRounds,
      selected.internal round.val % goldilocksP ≠ 0) ∧
    (∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      selected.terminal round.val lane % goldilocksP ≠ 0) := by
  decide

private theorem lcEval_basis_addConstant_ne
    (constant : Nat) (comb : Poseidon2Core.LinComb) (target : Nat)
    (notWire : target ≠ 0) :
    lcEval (basisAssignment target) (addConstant constant comb) =
      lcEval (basisAssignment target) comb := by
  have zeroNe : 0 ≠ target := Ne.symm notWire
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum, addConstant, rawSum_cons]
  simp [basisAssignment, zeroNe]

private theorem lcEval_basis_addConstant_wire
    (constant : Nat) (comb : Poseidon2Core.LinComb)
    (fresh : ¬ Mentions comb 0) :
    lcEval (basisAssignment 0) (addConstant constant comb) =
      constant % goldilocksP := by
  rw [lcEval_eq_rawSum, addConstant, rawSum_cons, basisAssignment, if_pos rfl,
    Nat.mul_one, rawSum_basis_not_mentions comb 0 fresh, Nat.add_zero]

theorem normalized_addConstant_nonzero
    (constant : Nat) (comb : Poseidon2Core.LinComb)
    (fresh : ¬ Mentions comb 0)
    (constantNonzero : constant % goldilocksP ≠ 0)
    (combNonzero :
      ∀ entry ∈ normalize comb, entry.2 % goldilocksP ≠ 0)
    (entry : Nat × Nat) (member : entry ∈ normalize (addConstant constant comb)) :
    entry.2 % goldilocksP ≠ 0 := by
  have readAdded :=
    lcEval_basis_normalized_entry (addConstant constant comb) entry member
  by_cases isWire : entry.1 = 0
  · rw [isWire] at readAdded
    rw [lcEval_basis_addConstant_wire constant comb fresh] at readAdded
    intro vanishes
    apply constantNonzero
    rw [readAdded, vanishes]
  · have mentionedAdded : Mentions (addConstant constant comb) entry.1 :=
      (mentions_normalize _ _).1
        (List.mem_map.2 ⟨entry, member, rfl⟩)
    have mentionedComb : Mentions comb entry.1 :=
      ((mentions_addConstant constant comb entry.1).1 mentionedAdded).resolve_left
        isWire
    have mentionedNormal : Mentions (normalize comb) entry.1 :=
      (mentions_normalize comb entry.1).2 mentionedComb
    rcases List.mem_map.1 mentionedNormal with ⟨source, sourceMember, sourceColumn⟩
    have readSource :=
      lcEval_basis_normalized_entry comb source sourceMember
    have evalSame :=
      lcEval_basis_addConstant_ne constant comb entry.1 isWire
    have columns : source.1 = entry.1 := sourceColumn
    rw [columns] at readSource
    rw [evalSame, readSource] at readAdded
    exact fun vanishes => combNonzero source sourceMember
      (by rw [readAdded, vanishes])

theorem fieldNormalize_addConstant_length
    (constant : Nat) (comb : Poseidon2Core.LinComb)
    (fresh : ¬ Mentions comb 0)
    (constantNonzero : constant % goldilocksP ≠ 0)
    (combNonzero :
      ∀ entry ∈ normalize comb, entry.2 % goldilocksP ≠ 0) :
    (fieldNormalize (addConstant constant comb)).length =
      (normalize comb).length + 1 := by
  rw [fieldNormalize_length_of_nonzero _
      (normalized_addConstant_nonzero constant comb fresh constantNonzero
        combNonzero),
    normalize_addConstant_length constant comb fresh]

/-! ## The seven concrete first-round entries -/

def firstRoundLength (call : Nat) (lane : Fin width) : Nat :=
  (fieldNormalize
    (addConstant (selected.initial 0 lane)
      (initialStateFrom (layout.call call)
        (entryOf layout chunkLength call) 0 lane))).length

def expectedFirstRoundLength (call : Nat) : Nat :=
  match call with
  | 0 => 5
  | 1 | 2 | 3 | 4 => 13
  | 5 => 12
  | 6 => 9
  | _ => 0

set_option maxRecDepth 100000 in
/-- The call-dependent boundary is only 7×8 combinations, each with at most
thirteen entries.  This certificate checks the actual concrete layouts,
padding merge, matrix coefficients, and selected round constants. -/
theorem selected_first_round_lengths :
    ∀ call : Fin calls, ∀ lane : Fin width,
      firstRoundLength call.val lane = expectedFirstRoundLength call.val := by
  decide

/-! ## Symbolic discharge of every other round -/

private theorem filterMap_some_of_length_eq
    {α β : Type} (function : α → Option β) :
    ∀ items : List α,
      (items.filterMap function).length = items.length →
      ∀ item ∈ items, ∃ value, function item = some value := by
  intro items equal item member
  induction items with
  | nil => simp at member
  | cons head tail hypothesis =>
      cases output : function head with
      | none =>
          simp only [List.filterMap_cons_none output, List.length_cons] at equal
          have bound := List.length_filterMap_le function tail
          omega
      | some value =>
          simp only [List.filterMap_cons_some output, List.length_cons] at equal
          rcases List.mem_cons.1 member with rfl | inTail
          · exact ⟨value, output⟩
          · exact hypothesis (Nat.add_right_cancel equal) inTail

private theorem normalized_nonzero_of_field_length
    (comb : Poseidon2Core.LinComb)
    (lengthEq :
      (fieldNormalize comb).length = (normalize comb).length) :
    ∀ entry ∈ normalize comb, entry.2 % goldilocksP ≠ 0 := by
  intro entry member
  unfold fieldNormalize at lengthEq
  rcases filterMap_some_of_length_eq reduceTerm (normalize comb) lengthEq
      entry member with ⟨value, kept⟩
  unfold reduceTerm at kept
  split at kept
  · simp at kept
  · assumption

def scheduledFieldSize (call index : Nat) : Nat :=
  if index < width then expectedFirstRoundLength call
  else scheduledSize index

def fieldScheduledSizes (call : Nat) : List Nat :=
  (List.finRange sboxCount).map
    (fun index =>
      (fieldNormalize
        (scheduleOfFrom (layout.call call)
          (entryOf layout chunkLength call) selected index)).length)

theorem scheduledFieldSize_sum_check :
    ∀ call : Fin calls,
      ((List.finRange sboxCount).map
        (fun index => scheduledFieldSize call.val index.val)).sum =
      1109 + 8 * expectedFirstRoundLength call.val := by
  decide

theorem fieldScheduledSizes_pointwise
    (call : Fin calls) (index : Fin sboxCount) :
    (fieldNormalize
      (scheduleOfFrom (layout.call call.val)
        (entryOf layout chunkLength call.val) selected index)).length =
      scheduledFieldSize call.val index.val := by
  have indexLt := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  unfold scheduledFieldSize
  by_cases firstRound : index.val < width
  · rw [if_pos firstRound]
    have laneLt : index.val < width := firstRound
    let lane : Fin width := ⟨index.val, laneLt⟩
    have indexEq : index.val = initialSboxIndex 0 lane.val := by
      simp [initialSboxIndex, lane]
    rw [scheduleOfFrom_initial (layout.call call.val)
      (entryOf layout chunkLength call.val) selected index 0 lane
      indexEq (by decide)]
    change firstRoundLength call.val lane = expectedFirstRoundLength call.val
    exact selected_first_round_lengths call lane
  · rw [if_neg firstRound]
    by_cases isInitial : index.val < halfFullRounds * width
    · have laneLt : index.val % width < width := Nat.mod_lt _ (by decide)
      let lane : Fin width := ⟨index.val % width, laneLt⟩
      let round := index.val / width
      have roundPositive : 1 ≤ round := by
        unfold round
        simp only [width] at firstRound ⊢
        omega
      have roundLt : round < halfFullRounds := by
        unfold round
        simp only [halfFullRounds, width] at isInitial ⊢
        omega
      have indexEq : index.val = initialSboxIndex round lane.val := by
        unfold round lane initialSboxIndex
        rw [Nat.mul_comm]
        exact (Nat.div_add_mod index.val width).symm
      rw [scheduleOfFrom_initial (layout.call call.val)
        (entryOf layout chunkLength call.val) selected index round lane
        indexEq roundLt]
      obtain ⟨previous, roundEq⟩ : ∃ previous, round = previous + 1 :=
        ⟨round - 1, by omega⟩
      rw [roundEq,
        initialStateFrom_succ_entry_irrelevant (layout.call call.val)
          (entryOf layout chunkLength call.val)
          (fun source => [((layout.call call.val).inputPort source, 1)])
          previous,
        ← initialState_eq_from (layout.call call.val) (previous + 1)]
      have stateField :=
        initialState_fieldNormalize_length (layout.call call.val)
          (layout_wellFormed.perCall call.val) (previous + 1) lane
      have stateNormal :=
        initialState_normalize_length (layout.call call.val)
          (layout_wellFormed.perCall call.val) (previous + 1) lane
      rw [fieldNormalize_addConstant_length
        (selected.initial (previous + 1) lane)
        (initialState (layout.call call.val) (previous + 1) lane)
        (initialState_not_mentions_zero (layout.call call.val)
          (layout_wellFormed.perCall call.val) (previous + 1) lane)
        (selected_constants_nonzero.1
          ⟨previous + 1, by simpa [roundEq] using roundLt⟩ lane)
        (normalized_nonzero_of_field_length _ (by
          rw [stateField, stateNormal]))]
      rw [stateNormal]
      unfold scheduledSize
      rw [if_pos (by
        simpa only [halfFullRounds, width] using isInitial)]
      decide
    · rw [scheduleOfFrom_nonInitial (layout.call call.val)
        (entryOf layout chunkLength call.val) selected index isInitial]
      by_cases isPartial :
          index.val < halfFullRounds * width + partialRounds
      · let round := index.val - halfFullRounds * width
        have roundLt : round < partialRounds := by
          unfold round
          omega
        have indexEq : index.val = halfFullRounds * width + round := by
          unfold round
          omega
        rw [Poseidon2Support.scheduleOf_partial (layout.call call.val) selected
          index round (by simpa only [halfFullRounds, width] using indexEq)
          roundLt]
        unfold partialSboxInput
        rw [fieldNormalize_addConstant_length (selected.internal round)
          (partialState (layout.call call.val) round ⟨0, by decide⟩)
          (partialState_not_mentions_zero (layout.call call.val) round _)
          (selected_constants_nonzero.2.1 ⟨round, roundLt⟩)
          (partialState_normalized_coefficients_nonzero
            (layout.call call.val) round (Nat.le_of_lt roundLt)
            ⟨0, by decide⟩),
          partialState_normalize_length]
        unfold scheduledSize
        rw [if_neg (by
          simp only [halfFullRounds, width] at isInitial
          omega), if_pos (by
            simp only [halfFullRounds, width, partialRounds] at isPartial
            omega)]
        have indexEqConcrete : index.val = 32 + round := by
          simpa only [halfFullRounds, width] using indexEq
        simp only [width]
        omega
      · let offset := index.val - (halfFullRounds * width + partialRounds)
        have laneLt : offset % width < width := Nat.mod_lt _ (by decide)
        let lane : Fin width := ⟨offset % width, laneLt⟩
        let round := offset / width
        have roundLt : round < halfFullRounds := by
          unfold round offset
          simp only [halfFullRounds, width, partialRounds] at indexLt ⊢
          omega
        have indexEq : index.val = terminalSboxIndex round lane.val := by
          have base : 54 ≤ index.val := by
            simp only [halfFullRounds, width, partialRounds] at isPartial
            omega
          have divMod := Nat.div_add_mod (index.val - 54) 8
          unfold terminalSboxIndex round lane offset
          simp only [halfFullRounds, width, partialRounds]
          omega
        rw [scheduleOf_terminal (layout.call call.val) selected index
          round lane indexEq roundLt, terminalSboxInput]
        cases roundCase : round with
        | zero =>
          have offsetLt : offset < width := by
            unfold round at roundCase
            simp only [width] at roundCase ⊢
            omega
          have indexBelow62 : index.val < 62 := by
            unfold offset at offsetLt
            simp only [halfFullRounds, width, partialRounds] at offsetLt isPartial
            omega
          simp only [terminalState]
          rw [fieldNormalize_addConstant_length (selected.terminal 0 lane)
            (partialState (layout.call call.val) partialRounds lane)
            (partialState_not_mentions_zero (layout.call call.val)
              partialRounds lane)
            (selected_constants_nonzero.2.2 ⟨0, by decide⟩ lane)
            (partialState_normalized_coefficients_nonzero
              (layout.call call.val) partialRounds (Nat.le_refl _) lane),
            partialState_normalize_length]
          unfold scheduledSize
          simp only [halfFullRounds, width, partialRounds] at isInitial isPartial
          rw [if_neg (by omega), if_neg (by omega), if_pos indexBelow62]
          decide
        | succ previous =>
          have offsetGe : width ≤ offset := by
            unfold round at roundCase
            simp only [width] at roundCase ⊢
            omega
          have indexGe62 : 62 ≤ index.val := by
            unfold offset at offsetGe
            simp only [halfFullRounds, width, partialRounds] at offsetGe isPartial
            omega
          have stateField :=
            terminalState_succ_fieldNormalize_length
              (layout.call call.val) previous lane
          have stateNormal :=
            terminalState_succ_normalize_length
              (layout.call call.val) previous lane
          rw [fieldNormalize_addConstant_length
            (selected.terminal (previous + 1) lane)
            (terminalState (layout.call call.val) (previous + 1) lane)
            (terminalState_succ_not_mentions_zero
              (layout.call call.val) previous lane)
            (selected_constants_nonzero.2.2
              ⟨previous + 1, by simpa [roundCase] using roundLt⟩ lane)
            (normalized_nonzero_of_field_length _ (by
              rw [stateField, stateNormal])),
            terminalState_succ_normalize_length]
          unfold scheduledSize
          simp only [halfFullRounds, width, partialRounds] at isInitial isPartial
          rw [if_neg (by omega), if_neg (by omega), if_neg (by omega)]
          decide

theorem fieldScheduledSizes_sum (call : Fin calls) :
    (fieldScheduledSizes call.val).sum =
      1109 + 8 * expectedFirstRoundLength call.val := by
  unfold fieldScheduledSizes
  rw [show
      (fun index : Fin sboxCount =>
        (fieldNormalize
          (scheduleOfFrom (layout.call call.val)
            (entryOf layout chunkLength call.val) selected index)).length) =
      (fun index => scheduledFieldSize call.val index.val) from
      funext (fieldScheduledSizes_pointwise call)]
  exact scheduledFieldSize_sum_check call

theorem all_fieldScheduledSizes_sum :
    ((List.finRange calls).map
      (fun call => (fieldScheduledSizes call.val).sum)).sum = 8387 := by
  rw [show
      (fun call : Fin calls => (fieldScheduledSizes call.val).sum) =
      (fun call => 1109 + 8 * expectedFirstRoundLength call.val) from
      funext fieldScheduledSizes_sum]
  decide

/-! ## Receipt-folded emitted coefficient total -/

private theorem fieldNormalize_singleton_length (column : Nat) :
    (fieldNormalize [(column, 1)]).length = 1 := by
  rfl

theorem normalizedSboxRows_termCount (frame : SboxFrame) :
    rawProgramTermCount (normalizeProgram (sboxRows frame)) =
      3 * (fieldNormalize frame.input).length + 9 := by
  simp only [rawProgramTermCount, normalizeProgram, sboxRows, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, rawTermCount, normalizeRow,
    rowSquare, rowFourth, rowSixth, rowSeventh, fieldNormalize_singleton_length]
  omega

theorem normalizedBindRow_termCount
    (comb : Poseidon2Core.LinComb) (port : Nat) :
    rawProgramTermCount (normalizeProgram [bindRow comb port]) =
      (fieldNormalize comb).length + 2 := by
  simp only [rawProgramTermCount, normalizeProgram, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, rawTermCount, normalizeRow, bindRow,
    fieldNormalize_singleton_length]
  omega

private theorem rawProgramTermCount_append (left right : List Row) :
    rawProgramTermCount (left ++ right) =
      rawProgramTermCount left + rawProgramTermCount right := by
  simp [rawProgramTermCount, List.map_append, List.sum_append]

private theorem rawProgramTermCount_flatMap {α : Type}
    (items : List α) (rows : α → List Row) :
    rawProgramTermCount (items.flatMap rows) =
      (items.map (fun item => rawProgramTermCount (rows item))).sum := by
  induction items with
  | nil => simp [rawProgramTermCount]
  | cons head tail hypothesis =>
      simp [rawProgramTermCount_append, hypothesis]

private theorem normalizeProgram_append (left right : List Row) :
    normalizeProgram (left ++ right) =
      normalizeProgram left ++ normalizeProgram right := by
  simp [normalizeProgram]

private theorem normalizeProgram_flatMap {α : Type}
    (items : List α) (rows : α → List Row) :
    normalizeProgram (items.flatMap rows) =
      items.flatMap (fun item => normalizeProgram (rows item)) := by
  simp [normalizeProgram, List.map_flatMap]

private theorem normalizedSboxProgram_termCount
    (call : Nat) :
    rawProgramTermCount
      (normalizeProgram
        (sboxProgram (layout.call call)
          (scheduleOfFrom (layout.call call)
            (entryOf layout chunkLength call) selected))) =
      3 * (fieldScheduledSizes call).sum + 9 * sboxCount := by
  unfold sboxProgram fieldScheduledSizes
  rw [normalizeProgram_flatMap, rawProgramTermCount_flatMap]
  simp only [normalizedSboxRows_termCount]
  exact sum_map_three_plus_nine _ _

private theorem normalizedBindingProgram_termCount (call : Nat) :
    rawProgramTermCount
      (normalizeProgram
        (bindingProgram (layout.call call) (finalState (layout.call call)))) =
      80 := by
  unfold bindingProgram terminalBindingRows
  unfold normalizeProgram rawProgramTermCount
  rw [List.map_map, List.map_map]
  simp only [Function.comp_def, rawTermCount, normalizeRow, bindRow,
    fieldNormalize_singleton_length]
  rw [show
      (fun lane : Fin width =>
        (fieldNormalize (finalState (layout.call call) lane)).length + 1 + 1) =
      (fun _lane : Fin width => 10) from
      funext (fun lane => by
        have finalLength :=
          finalState_fieldNormalize_length (layout.call call) lane
        simp only [width] at finalLength
        omega)]
  decide

theorem callProgram_termCount (call : Fin calls) :
    rawProgramTermCount
      (normalizedCanonicalProgramFrom (layout.call call.val)
        (entryOf layout chunkLength call.val) selected) =
      3 * (fieldScheduledSizes call.val).sum + 854 := by
  unfold normalizedCanonicalProgramFrom canonicalProgramFrom permutationProgram
  rw [normalizeProgram_append, rawProgramTermCount_append,
    normalizedSboxProgram_termCount, normalizedBindingProgram_termCount]
  simp only [sboxCount, externalRounds, width, partialRounds]

def expectedCallCoefficientCount (call : Nat) : Nat :=
  3 * (1109 + 8 * expectedFirstRoundLength call) + 854

theorem expected_call_coefficient_count_sum :
    ((List.range calls).map expectedCallCoefficientCount).sum = 31139 := by
  decide

theorem program_nonzero_coefficient_count :
    rawProgramTermCount (program selected) = 31139 := by
  unfold program spongeProgram
  rw [rawProgramTermCount_flatMap]
  have callCounts :
      (List.range calls).map
          (fun call =>
            rawProgramTermCount
              (normalizedCanonicalProgramFrom (layout.call call)
                (entryOf layout chunkLength call) selected)) =
        (List.range calls).map
          (fun call => 3 * (fieldScheduledSizes call).sum + 854) := by
    apply List.map_congr_left
    intro call member
    exact callProgram_termCount ⟨call, List.mem_range.1 member⟩
  rw [callCounts]
  have expectedCounts :
      (List.range calls).map
          (fun call => 3 * (fieldScheduledSizes call).sum + 854) =
        (List.range calls).map expectedCallCoefficientCount := by
    apply List.map_congr_left
    intro call member
    rw [fieldScheduledSizes_sum ⟨call, List.mem_range.1 member⟩]
    rfl
  rw [expectedCounts]
  exact expected_call_coefficient_count_sum

end Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients
