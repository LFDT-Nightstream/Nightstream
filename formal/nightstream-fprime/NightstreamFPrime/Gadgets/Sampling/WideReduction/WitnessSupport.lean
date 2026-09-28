import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Gadgets.Range.CanonicalU64.Witness

/-! Read bounds for the wide sampler's helper hints and checked witness.
Parents use this contract without unfolding the sampler's child circuits. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Circuit

theorem witnesses_main_readsSatisfy (interface : Interface) (hints : Nat → List Hint)
    (hintCount : ∀ offset, (hints offset).length = newBitCount)
    (offset : Nat) (allowed : Nat → Prop)
    (sources : ∀ lane, (interface.source lane offset).VarsSatisfy allowed)
    (locals : ∀ index, index < privateCount → allowed (offset + index))
    (hintReads : ∀ hint ∈ hints offset, hint.source.VarsSatisfy allowed) :
    ∀ batch ∈ witnesses (Circuit.ops (circuit interface hints hintCount).main offset),
      batch.ReadsSatisfy allowed := by
  intro batch member
  change batch ∈ witnesses (operations interface hints offset) at member
  simp only [operations, witnesses, List.flatMap_append, List.mem_append] at member
  rcases member with (child | hinted) | row
  · rcases List.mem_flatMap.mp child with ⟨op, opMember, batchMember⟩
    rcases List.mem_map.mp opMember with ⟨lane, _, rfl⟩
    change batch ∈ witnesses (Circuit.ops
      (Range.CanonicalU64.main (childInterface interface offset lane))
      (childOffset offset lane)) at batchMember
    rw [Range.CanonicalU64.witnesses_main] at batchMember
    apply Range.CanonicalU64.witnessBatches_readsSatisfy _ _ allowed
      (sources lane) _ batch batchMember
    intro index indexLt
    have laneLt : lane.val < 4 := lane.isLt
    change allowed (offset + childWidth * lane.val + index)
    rw [Nat.add_assoc]
    apply locals
    norm_num [privateCount, childWidth, fieldCount, newBitCount,
      quotientBitCount, digitCount, digitBitCount, checkCount, checkBitCount,
      Range.CanonicalU64.auxiliaryCount] at indexLt ⊢
    omega
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
      Op.witnesses, List.mem_singleton] at hinted
    subst batch
    exact (WitnessBatch.readsSatisfy_hinted allowed _ _).mpr hintReads
  · simp [rowOps, List.flatMap_append, List.flatMap_map, Op.witnesses] at row

namespace Program

private theorem helpers_below (interface : Interface) (start : Nat)
    (sources : ∀ lane, (interface.source lane start).VarsBelow (start + privateCount)) :
    ∀ hint ∈ HintProgram.helpers interface start,
      hint.source.VarsBelow (start + privateCount) := by
  intro hint member
  simp only [HintProgram.helpers, List.mem_append] at member
  rcases member with (source | limb) | division
  · obtain ⟨lane, _, member⟩ := List.mem_flatMap.mp source
    obtain ⟨bit, _, rfl⟩ := List.mem_map.mp member
    exact sources lane
  · obtain ⟨position, positionMember, member⟩ := List.mem_flatMap.mp limb
    have positionLt := List.mem_range.mp positionMember
    simp only [HintProgram.limbHints, List.mem_cons] at member
    rcases member with rfl | member
    · refine ⟨trivial, Expr.VarsBelow.mono _ (HintSupport.limbTerms_below start position) ?_⟩
      norm_num [HintProgram.limbStart, HintProgram.sourceBitCount, HintProgram.limbStride,
        HintProgram.accumulatorBits, HintProgram.limbCount, privateCount_eq] at *
      omega
    · obtain ⟨bit, _, rfl⟩ := List.mem_map.mp member
      change HintProgram.limbStart start position < start + privateCount
      norm_num [HintProgram.limbStart, HintProgram.sourceBitCount, HintProgram.limbStride,
        HintProgram.accumulatorBits, HintProgram.limbCount, privateCount_eq] at *
      omega
  · obtain ⟨round, roundMember, member⟩ := List.mem_flatMap.mp division
    obtain ⟨position, positionMember, member⟩ := List.mem_flatMap.mp member
    have roundLt := List.mem_range.mp roundMember
    have positionLt := List.mem_range.mp positionMember
    have source := Expr.VarsBelow.mono _ (HintSupport.divisionInput_below start round position)
      (show HintProgram.divisionColumn start round position ≤ start + privateCount by
        norm_num [HintProgram.divisionColumn, HintProgram.divisionStart,
          HintProgram.sourceBitCount, HintProgram.limbCount, HintProgram.limbStride,
          HintProgram.accumulatorBits, HintProgram.divisionStride, HintProgram.divisionLimbs,
          digitCount, privateCount_eq] at *
        omega)
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> exact source

private theorem resultHints_below (start : Nat) :
    ∀ hint ∈ HintProgram.resultHints start (coreOffset start),
      hint.source.VarsBelow (start + privateCount) := by
  intro hint member
  simp only [HintProgram.resultHints, List.mem_append] at member
  rcases member with (quotient | digit) | check
  · obtain ⟨bit, _, rfl⟩ := List.mem_map.mp quotient
    change HintProgram.divisionColumn start (digitCount - 1)
      (HintProgram.divisionLimbs - 1 - bit / HintProgram.divisionBits) < start + privateCount
    norm_num [HintProgram.divisionColumn, HintProgram.divisionStart,
      HintProgram.sourceBitCount, HintProgram.limbCount, HintProgram.limbStride,
      HintProgram.accumulatorBits, HintProgram.divisionStride, HintProgram.divisionLimbs,
      digitCount, privateCount_eq, HintProgram.divisionBits]
    omega
  · obtain ⟨position, positionMember, member⟩ := List.mem_flatMap.mp digit
    obtain ⟨bit, _, rfl⟩ := List.mem_map.mp member
    have positionLt := List.mem_range.mp positionMember
    change HintProgram.divisionColumn start position (HintProgram.divisionLimbs - 1) + 1 < _
    norm_num [HintProgram.divisionColumn, HintProgram.divisionStart,
      HintProgram.sourceBitCount, HintProgram.limbCount, HintProgram.limbStride,
      HintProgram.accumulatorBits, HintProgram.divisionStride, HintProgram.divisionLimbs,
      digitCount, privateCount_eq] at *
    omega
  · obtain ⟨position, _, member⟩ := List.mem_flatMap.mp check
    obtain ⟨bit, _, rfl⟩ := List.mem_map.mp member
    apply Expr.VarsBelow.mono _ (HintSupport.checkQuotient_below (coreOffset start) position)
    norm_num [checkStart, digitStart, quotientStart, coreOffset, HintProgram.helperCount_eq,
      childWidth, Range.CanonicalU64.auxiliaryCount, fieldCount, quotientBitCount,
      digitCount, digitBitCount, privateCount_eq]

theorem witnesses_main_below (interface : Interface) (start : Nat)
    (sources : ∀ lane, (interface.source lane start).VarsBelow (start + privateCount)) :
    ∀ batch ∈ witnesses (Circuit.ops (circuit interface).main start),
      batch.ReadsSatisfy (fun column => column < start + privateCount) := by
  intro batch member
  change batch ∈ witnesses (operations interface start) at member
  simp only [operations, witnesses, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, Op.witnesses, List.mem_append, List.mem_singleton] at member
  rcases member with rfl | member
  · rw [WitnessBatch.readsSatisfy_hinted]
    intro hint member
    exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr (helpers_below interface start sources hint member)
  · refine WideReduction.witnesses_main_readsSatisfy (coreInterface interface start)
      (HintProgram.resultHints start) (HintProgram.resultHints_length start)
      (coreOffset start) _ ?_ ?_ ?_ batch member
    · intro lane
      exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr (sources lane)
    · intro index below
      change coreOffset start + index < start + privateCount
      simp only [coreOffset, privateCount]
      omega
    · intro hint member
      exact (Expr.varsSatisfy_lt_iff_varsBelow _ _).mpr (resultHints_below start hint member)

end Program
end NightstreamFPrime.Gadgets.Sampling.WideReduction
