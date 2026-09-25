import NightstreamFPrime.Export.Stage1.PiRLCCombinationReadSupport
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations

/-! Canonical First54 inputs and local allocations precede product scratch. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54ScratchGeometry

open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Export.Package
open PiRLCFirst54Invocations
open PiRLCCombinationScratchGeometry (scratchStart)

private theorem source_local (column : Nat) (lower : PiRLCStarts.phaseLogicalStart ≤ column) :
    Spartan.piCcsPhaseOffset ≤ column := by
  change 19513117 ≤ column at lower
  change 14751804 ≤ column
  omega

private theorem mapped_add_before (column offset : Nat)
    (lower : PiRLCStarts.phaseLogicalStart ≤ column)
    (upper : column + offset < PiRLCStarts.commitmentFreshStart) :
    finalColumn column + offset < scratchStart := by
  unfold finalColumn
  rw [← Spartan.sourceToSpartan_add_of_piCcsLocal column offset (source_local column lower)]
  exact Spartan.sourceToSpartan_lt_of_piCcsLocal _ _
    (by have := source_local column lower; omega) upper

private theorem decoder_before (source round extra offset : Nat)
    (sourceBound : source < 17) (roundBound : round < 64) (extraBound : extra ≤ 16)
    (offsetBound : offset < 1) :
    finalColumn (decoderLogicalStart source round + extra) + offset < scratchStart := by
  apply mapped_add_before
  all_goals
    norm_num [decoderLogicalStart, candidateDigestRound, candidateLane, candidatePart,
      PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart_eq,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      CanonicalU64.auxiliaryCount, Candidate16Five.auxiliaryCount,
      PiRLCStarts.commitmentFreshStart_eq]
    omega

private theorem position_before (source round slot offset : Nat)
    (sourceBound : source < 17) (roundBound : round < 64) (positionBound : slot + offset < 55) :
    finalColumn (positionSourceStart source round + slot) + offset < scratchStart := by
  apply mapped_add_before
  all_goals
    norm_num [positionSourceStart, First54.positionOffset, First54.roundPrivateCount,
      First54Step.slotCount, First54ValueStep.outputCount,
      PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart_eq,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      PiRLCStarts.commitmentFreshStart_eq]
    omega

private theorem value_before (source round slot offset : Nat)
    (sourceBound : source < 17) (roundBound : round < 64) (positionBound : slot + offset < 54) :
    finalColumn (valueSourceStart source round + slot) + offset < scratchStart := by
  apply mapped_add_before
  all_goals
    norm_num [valueSourceStart, First54.valueOffset, First54.positionOffset,
      First54.roundPrivateCount, First54Step.slotCount, First54ValueStep.outputCount,
      PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart_eq,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      PiRLCStarts.commitmentFreshStart_eq]
    omega

private def RangesBefore (ranges : List CompactInputRange) : Prop :=
  ∀ range ∈ ranges, ∀ offset, offset < range.inputCount →
    range.columnStart + offset * range.columnStride < scratchStart

private theorem position_ranges_before (source round : Nat) (slot : Fin First54Step.slotCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    RangesBefore (positionInvocation source round slot.val).inputRanges := by
  intro range member offset offsetBound
  have slotBound : slot.val < 55 := slot.isLt
  simp only [positionInvocation] at member
  split_ifs at member with first
  · simp only [firstPositionInputRanges, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;>
      dsimp only at offsetBound ⊢ <;> simp only [Nat.mul_one]
    · exact decoder_before source round 16 offset sourceBound roundBound (by decide) offsetBound
    · exact position_before source round slot.val offset sourceBound roundBound (by
        change offset < 1 at offsetBound
        omega)
  · simp only [laterPositionInputRanges, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;>
      dsimp only at offsetBound ⊢ <;> simp only [Nat.mul_one]
    · exact decoder_before source round 16 offset sourceBound roundBound (by decide) offsetBound
    · simpa only [previousPositionSourceStart, Nat.add_zero, Nat.mul_one] using
        position_before source (round - 1) 0 offset sourceBound (by omega) (by simpa using! offsetBound)
    · exact position_before source round slot.val offset sourceBound roundBound (by
        change offset < 1 at offsetBound
        omega)

private theorem value_ranges_before (source round : Nat) (slot : Fin First54ValueStep.outputCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    RangesBefore (valueInvocation source round slot.val).inputRanges := by
  intro range member offset offsetBound
  have slotBound : slot.val < 54 := slot.isLt
  simp only [valueInvocation] at member
  split_ifs at member with first
  · simp only [firstValueInputRanges, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;>
      dsimp only at offsetBound ⊢ <;> simp only [Nat.mul_one]
    · exact decoder_before source round 16 offset sourceBound roundBound (by decide) offsetBound
    · exact decoder_before source round 1 offset sourceBound roundBound (by decide) offsetBound
    · exact value_before source round slot.val offset sourceBound roundBound (by
        change offset < 1 at offsetBound
        omega)
  · simp only [laterValueInputRanges, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl <;>
      dsimp only at offsetBound ⊢ <;> simp only [Nat.mul_one]
    · exact decoder_before source round 16 offset sourceBound roundBound (by decide) offsetBound
    · exact decoder_before source round 1 offset sourceBound roundBound (by decide) offsetBound
    · simpa only [previousPositionSourceStart, Nat.add_zero, Nat.mul_one] using
        position_before source (round - 1) 0 offset sourceBound (by omega) (by simpa using! offsetBound)
    · simpa only [previousValueSourceStart, Nat.add_zero, Nat.mul_one] using
        value_before source (round - 1) 0 offset sourceBound (by omega) (by simpa using! offsetBound)
    · exact value_before source round slot.val offset sourceBound roundBound (by
        change offset < 1 at offsetBound
        omega)

private theorem cached_column_eq (context : PerApplicationCachedShift.Context)
    (column : Nat) (before : column < scratchStart) : context.column column = column := by
  unfold PerApplicationCachedShift.Context.column
  apply if_pos
  change column < 28784740
  change column < 20572364 at before
  omega

private theorem shifted_ranges_before (context : PerApplicationCachedShift.Context)
    (ranges : List CompactInputRange) (before : RangesBefore ranges) :
    RangesBefore (ranges.map (PerApplicationCachedShift.shiftCompactInputRange context)) := by
  intro range member offset offsetBound
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  have bound := before source sourceMember offset offsetBound
  have columnBefore : source.columnStart < scratchStart := by omega
  simpa only [PerApplicationCachedShift.shiftCompactInputRange,
    cached_column_eq context source.columnStart columnBefore] using bound

private theorem input_before (ranges : List CompactInputRange)
    (before : RangesBefore ranges) (input : Nat) : compactInputColumn ranges input < scratchStart := by
  unfold compactInputColumn
  cases found : ranges.find? (fun range =>
      range.inputStart ≤ input ∧ input < range.inputStart + range.inputCount) with
  | none =>
      change 0 < 20572364
      decide
  | some range =>
      have member := List.mem_of_find?_eq_some found
      have interval : range.inputStart ≤ input ∧ input < range.inputStart + range.inputCount := by
        simpa using List.find?_some found
      exact before range member (input - range.inputStart) (by omega)

theorem position_inputs_before (context : PerApplicationCachedShift.Context)
    (source round : Nat) (slot : Fin First54Step.slotCount)
    (sourceBound : source < 17) (roundBound : round < 64) (input : Nat) :
    compactInputColumn (PerApplicationCachedShift.shiftCompactRowInvocation context
      (positionInvocation source round slot.val)).inputRanges input < scratchStart :=
  input_before _ (shifted_ranges_before context _
    (position_ranges_before source round slot sourceBound roundBound)) input

theorem value_inputs_before (context : PerApplicationCachedShift.Context)
    (source round : Nat) (slot : Fin First54ValueStep.outputCount)
    (sourceBound : source < 17) (roundBound : round < 64) (input : Nat) :
    compactInputColumn (PerApplicationCachedShift.shiftCompactRowInvocation context
      (valueInvocation source round slot.val)).inputRanges input < scratchStart :=
  input_before _ (shifted_ranges_before context _
    (value_ranges_before source round slot sourceBound roundBound)) input

private theorem mapped_end_le (column count : Nat)
    (lower : PiRLCStarts.phaseLogicalStart ≤ column)
    (upper : column + count ≤ PiRLCStarts.commitmentFreshStart) :
    finalColumn column + count ≤ scratchStart := by
  have localBound := source_local column lower
  have decompose : column = Spartan.piCcsPhaseOffset + (column - Spartan.piCcsPhaseOffset) := by omega
  unfold finalColumn
  rw [decompose, Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (Nat.le_refl _)]
  change 14751526 + (column - 14751804) + count ≤ 20572364
  change 14751804 ≤ column at localBound
  change column + count ≤ 20572642 at upper
  omega

private theorem position_local_end (source round : Nat) (slot : Fin First54Step.slotCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    (positionInvocation source round slot.val).localStart + 6 ≤ scratchStart := by
  rw [positionInvocation_localStart]
  apply mapped_end_le
  all_goals
    have slotBound : slot.val < 55 := slot.isLt
    cases round <;>
      norm_num [PiRLCStarts.selectorFreshStart, PiRLCStarts.samplerSourceFreshStart,
        PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
        PiRLCStarts.phaseLogicalStart_eq, PiRLCStarts.commitmentFreshStart_eq,
        roundFreshPrefix, positionFreshPrefix] at roundBound ⊢
  all_goals (try split_ifs) <;> omega

private theorem value_local_end (source round : Nat) (slot : Fin First54ValueStep.outputCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    (valueInvocation source round slot.val).localStart + 4 ≤ scratchStart := by
  rw [valueInvocation_localStart]
  apply mapped_end_le
  all_goals
    have slotBound : slot.val < 54 := slot.isLt
    cases round <;>
      norm_num [PiRLCStarts.selectorFreshStart, PiRLCStarts.samplerSourceFreshStart,
        PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
        PiRLCStarts.phaseLogicalStart_eq, PiRLCStarts.commitmentFreshStart_eq,
        roundFreshPrefix, valueFreshPrefix, positionFreshCount] at roundBound ⊢ <;> omega

theorem shifted_position_local_end (context : PerApplicationCachedShift.Context)
    (source round : Nat) (slot : Fin First54Step.slotCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    (PerApplicationCachedShift.shiftCompactRowInvocation context
      (positionInvocation source round slot.val)).localStart + 6 ≤ scratchStart := by
  have bound := position_local_end source round slot sourceBound roundBound
  change context.column _ + 6 ≤ scratchStart
  rw [cached_column_eq context _ (by omega)]
  exact bound

theorem shifted_value_local_end (context : PerApplicationCachedShift.Context)
    (source round : Nat) (slot : Fin First54ValueStep.outputCount)
    (sourceBound : source < 17) (roundBound : round < 64) :
    (PerApplicationCachedShift.shiftCompactRowInvocation context
      (valueInvocation source round slot.val)).localStart + 4 ≤ scratchStart := by
  have bound := value_local_end source round slot sourceBound roundBound
  change context.column _ + 4 ≤ scratchStart
  rw [cached_column_eq context _ (by omega)]
  exact bound

end NightstreamFPrime.Export.Stage1.PiRLCFirst54ScratchGeometry
