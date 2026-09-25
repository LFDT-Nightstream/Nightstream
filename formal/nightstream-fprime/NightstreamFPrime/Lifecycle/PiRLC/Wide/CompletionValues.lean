import NightstreamFPrime.Circuit.SequenceValues
import NightstreamFPrime.Gadgets.Sampling.WideReduction.ProgramValues
import NightstreamFPrime.Lifecycle.PiRLC.Wide.ProjectedBatch

/-! Exact range-program readback for the constructed wide sampler witness.
The completion condition is carried through the existing child sequence;
it is never inferred from row acceptance. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets NightstreamFPrime.Gadgets.Sampling

namespace Scalar

def RangeCompleted (interface : Interface) (coordinate offset : Nat) (env : Env) : Prop :=
  ∀ index, index < WideReduction.privateCount →
    env (WideReduction.Program.coreOffset (rangeOffset offset) + index) =
      WideReduction.Program.completeEnv (rangeInterface interface coordinate offset) env (rangeOffset offset)
        (WideReduction.Program.coreOffset (rangeOffset offset) + index)

theorem rangeCompleted_of_agree (interface : Interface) (coordinate offset : Nat)
    (inputs : Assumptions interface offset) (before after : Env)
    (completed : RangeCompleted interface coordinate offset before)
    (agrees : ∀ index, index < advanceOffset offset → after index = before index) :
    RangeCompleted interface coordinate offset after := by
  intro index bounded
  rw [agrees _ (by
    change index < 617 at bounded
    change offset + 592 + 1404 + index < offset + 592 + 2021
    omega), completed index bounded]
  apply WideReduction.ProgramValues.completeEnv_retained_congr _ _ _ _ _ _
    (range_inputs interface coordinate offset inputs) (range_inputs interface coordinate offset inputs)
    _ index bounded
  intro lane
  apply Expr.eval_eq_of_agree_below _ (rangeOffset offset) _ _
    (range_inputs interface coordinate offset inputs lane)
  intro column below
  exact (agrees column (by unfold advanceOffset; omega)).symm

/-- The scalar constructor returns its actual checked range-program values. -/
theorem complete_with_values (interface : Interface) (coordinate : Nat) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∃ completed,
      AgreesOutside env completed offset (localLength (operations interface coordinate offset)) ∧
      holdsFlat completed (operations interface coordinate offset) ∧
      RangeCompleted interface coordinate offset completed := by
  obtain ⟨entered, entryAgreement, entryRows⟩ :=
    v1_1.TranscriptAbsorption.complete interface coordinate env offset inputs
  obtain ⟨first, firstOps, firstEnd, _, _⟩ := Sequence.appendBuiltAt_current
    (Sequence.empty env offset) "pirlc.wide.enter_scalar" (entry interface coordinate) offset (by rfl)
    (entry_scope interface coordinate offset inputs) entered entryAgreement entryRows
  have firstStart : offset + localLength first.operations = rangeOffset offset := by
    simpa only [entry, FormalCircuit.withConstantFootprint_main,
      v1_1.TranscriptAbsorption.localLength_eq, rangeOffset] using firstEnd
  let range := rangeInterface interface coordinate offset
  let decoded := WideReduction.Program.completeEnv range first.current (rangeOffset offset)
  have rangeCorrect := WideReduction.Program.completeEnv_correct range first.current (rangeOffset offset)
    (range_inputs interface coordinate offset inputs)
  have rangeAgreement : AgreesOutside first.current decoded (rangeOffset offset)
      (localLength (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset))) := by
    change AgreesOutside first.current decoded _ (localLength (WideReduction.Program.operations _ _))
    rw [WideReduction.Program.localLength_eq]
    exact rangeCorrect.1
  obtain ⟨second, secondOps, secondEnd, _, secondCurrent⟩ := Sequence.appendBuiltAt_current first "pirlc.wide.reduce"
    (rangeCircuit interface coordinate offset) (rangeOffset offset) firstStart
    (range_scope interface coordinate offset inputs) decoded rangeAgreement rangeCorrect.2
  have secondStart : offset + localLength second.operations = advanceOffset offset := by
    change offset + localLength second.operations = rangeOffset offset + WideReduction.Program.privateCount
    have count := WideReduction.Program.localLength_eq range (rangeOffset offset)
    change localLength (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset)) =
      WideReduction.Program.privateCount at count
    rwa [count] at secondEnd
  obtain ⟨advanced, advanceAgreement, advanceRows⟩ :=
    Poseidon2.Permutation.Owned.complete (advanceInterface interface coordinate offset) second.current
      (advanceOffset offset) (advance_inputs interface coordinate offset inputs second.current)
  obtain ⟨third, thirdOps, _, _, thirdCurrent⟩ := Sequence.appendBuiltAt_current second "pirlc.wide.advance"
    (advance interface coordinate offset) (advanceOffset offset) secondStart
    (advance_scope interface coordinate offset inputs) advanced advanceAgreement advanceRows
  have allOps : third.operations = operations interface coordinate offset := by
    rw [thirdOps, secondOps, firstOps]
    rfl
  have preserved (index : Nat) (below : index < advanceOffset offset) : third.current index = decoded index := by
    rw [thirdCurrent, advanceAgreement index (Or.inl below), secondCurrent]
  refine ⟨third.current, by simpa only [allOps] using third.agrees,
    by simpa only [allOps] using third.rows, ?_⟩
  intro index bounded
  rw [preserved _ (by
    change index < 617 at bounded
    change offset + 592 + 1404 + index < offset + 592 + 2021
    omega)]
  apply WideReduction.ProgramValues.completeEnv_retained_congr _ _ _ _ _ _
    (range_inputs interface coordinate offset inputs) (range_inputs interface coordinate offset inputs)
    _ index bounded
  intro lane
  apply Expr.eval_eq_of_agree_below _ (rangeOffset offset) _ _
    (range_inputs interface coordinate offset inputs lane)
  intro column below
  exact ((preserved column (by unfold advanceOffset; omega)).trans
    (rangeCorrect.1 column (Or.inl below))).symm

end Scalar

namespace Batch

def RangesCompleted (interface : Interface) (offset count : Nat) (env : Env) : Prop :=
  ∀ source, source < count →
    Scalar.RangeCompleted (childInterface interface offset source) source (sourceOffset offset source) env

theorem rangesCompleted_of_agree (interface : Interface) (offset count : Nat)
    (inputs : Assumptions interface offset) (before after : Env)
    (completed : RangesCompleted interface offset count before)
    (agrees : ∀ index, index < offset + count * Scalar.privateCount → after index = before index) :
    RangesCompleted interface offset count after := by
  intro source bounded
  apply Scalar.rangeCompleted_of_agree _ _ _ (child_inputs interface offset source inputs)
    before after (completed source bounded)
  intro index below
  apply agrees
  change index < offset + source * 3205 + 2613 at below
  change index < offset + count * 3205
  omega

theorem complete_prefix_with_values (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (count : Nat) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = prefixOps interface offset count ∧
      RangesCompleted interface offset count completed.current := by
  induction count with
  | zero => exact ⟨Sequence.empty env offset, rfl, by intro source bound; omega⟩
  | succ count ih =>
    obtain ⟨before, beforeOps, beforeValues⟩ := ih
    obtain ⟨built, agreement, rows, values⟩ := Scalar.complete_with_values (childInterface interface offset count) count
      before.current (sourceOffset offset count) (child_inputs interface offset count inputs)
    obtain ⟨after, afterOps, _, preserved, current⟩ := Sequence.appendBuiltAt_current before (childName count)
      (Scalar.circuit (childInterface interface offset count) count) (sourceOffset offset count)
      (by rw [beforeOps, prefix_length]; rfl)
      (child_scope interface offset count inputs) built agreement rows
    refine ⟨after, by rw [afterOps, beforeOps, prefix_succ]; rfl, ?_⟩
    intro source bounded
    by_cases last : source = count
    · subst source
      rw [current]
      exact values
    · apply rangesCompleted_of_agree interface offset count inputs before.current after.current beforeValues
        (fun index below => preserved.values index (by rwa [beforeOps, prefix_length])) source (by omega)

theorem complete_with_values (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∃ completed, AgreesOutside env completed offset (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) ∧ RangesCompleted interface offset sourceCount completed := by
  obtain ⟨done, same, values⟩ := complete_prefix_with_values interface env offset inputs sourceCount
  exact ⟨done.current, by simpa only [operations, ← same] using done.agrees,
    by simpa only [operations, ← same] using done.rows, values⟩

end Batch

namespace ProjectedBatch

theorem complete_with_values (interface : Interface) (env : Env) (offset : Nat) (inputs : Assumptions interface offset) :
    ∃ completed, AgreesOutside env completed offset (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) ∧ Batch.RangesCompleted interface offset sourceCount completed := by
  obtain ⟨sampled, sampleAgreement, sampleRows, sampledValues⟩ := Batch.complete_with_values interface env offset inputs
  obtain ⟨before, beforeOps, _, _, beforeCurrent⟩ := Sequence.appendBuiltAt_current (Sequence.empty env offset)
    "pirlc.wide.batch" (Batch.circuit interface) offset (by rfl)
    (by rw [Batch.circuit_ops, Batch.localLength_eq]; exact Batch.scope interface offset inputs)
    sampled (by rw [Batch.circuit_ops]; exact sampleAgreement) (by rw [Batch.circuit_ops]; exact sampleRows)
  have beforeLength : localLength before.operations = Batch.privateCount := by
    rw [beforeOps]
    simp only [Sequence.empty, List.nil_append, localLength, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, Sequence.childOp_localLength, Nat.add_zero]
    change localLength (Circuit.ops (Batch.circuit interface).main offset) = _
    rw [Batch.circuit_ops, Batch.localLength_eq]
  obtain ⟨built, wordAgreement, wordRows⟩ := DigitWords.complete offset (wordsOffset offset) before.current (Nat.le_refl _)
  obtain ⟨done, doneOps, _, preserved, _⟩ := Sequence.appendBuiltAt_current before "pirlc.wide.digit_words"
    (DigitWords.circuit offset) (wordsOffset offset) (by rw [beforeLength]; rfl)
    (by rw [DigitWords.circuit_ops, DigitWords.localLength_eq]; exact DigitWords.scope offset (wordsOffset offset) (Nat.le_refl _))
    built (by rw [DigitWords.circuit_ops, DigitWords.localLength_eq]; exact wordAgreement)
    (by rw [DigitWords.circuit_ops]; exact wordRows)
  have same : done.operations = operations interface offset := by
    rw [doneOps, beforeOps]
    rfl
  refine ⟨done.current, by simpa only [same] using done.agrees, by simpa only [same] using done.rows, ?_⟩
  apply Batch.rangesCompleted_of_agree interface offset sourceCount inputs before.current done.current
  · rwa [beforeCurrent]
  · intro index below
    exact preserved.values index (by rwa [beforeLength])

end ProjectedBatch

end NightstreamFPrime.Lifecycle.PiRLC.Wide
