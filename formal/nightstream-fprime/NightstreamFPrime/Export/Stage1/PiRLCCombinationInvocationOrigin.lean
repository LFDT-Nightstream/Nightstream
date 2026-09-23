import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchGeometry
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift

/-!
Identifies the canonical product invocations from their template range.
The application shift preserves template selection and the product scratch
interval, and keeps every product input read outside that interval.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocationOrigin

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Package
open PackagePlan

private theorem first54_templateIndex_ge (block : First54InvocationBlock)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand) :
    108 ≤ invocation.templateIndex := by
  unfold First54InvocationBlock.expand at member
  rcases List.mem_flatMap.mp member with ⟨source, _sourceMember, roundMember⟩
  rcases List.mem_flatMap.mp roundMember with ⟨round, _roundMember, invocationMember⟩
  unfold PiRLCFirst54Invocations.roundInvocations at invocationMember
  rcases List.mem_append.mp invocationMember with positionMember | valueMember
  · unfold PiRLCFirst54Invocations.positionInvocations at positionMember
    rcases List.mem_map.mp positionMember with ⟨slot, _slotMember, rfl⟩
    simp only [PiRLCFirst54Invocations.positionInvocation,
      PiRLCFirst54Invocations.positionTemplateIndex,
      PiRLCFirst54Invocations.templateBase, PiRLCCombinationTemplates.templates_length]
    exact Nat.le_add_right _ _
  · unfold PiRLCFirst54Invocations.valueInvocations at valueMember
    rcases List.mem_map.mp valueMember with ⟨slot, _slotMember, rfl⟩
    simp only [PiRLCFirst54Invocations.valueInvocation,
      PiRLCFirst54Invocations.valueTemplateIndex,
      PiRLCFirst54Invocations.templateBase, PiRLCCombinationTemplates.templates_length]
    exact Nat.le_add_right _ _

private theorem combination_origin (invocation : CompactRowInvocation)
    (member : invocation ∈ canonicalCombinationBlock.expand) :
    ∃ descriptor : PiRLCProductSchedule.Descriptor,
      invocation = descriptor.compactInvocation := by
  rw [canonicalCombinationBlock_expand,
    ← PiRLCProductSchedule.compactInvocations_eq] at member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact ⟨PiRLCProductSchedule.descriptor index,
    congrFun PiRLCProductSchedule.compactInvocation_eq_descriptor index⟩

/-- No other canonical compact family uses the product template range. -/
theorem canonical_origin (block : CompactInvocationBlock)
    (blockMember : block ∈ canonicalCompactBlocks)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand)
    (productTemplate : invocation.templateIndex < 2 * ringDegree) :
    ∃ descriptor : PiRLCProductSchedule.Descriptor,
      invocation = descriptor.compactInvocation := by
  simp only [canonicalCompactBlocks, List.mem_cons, List.not_mem_nil, or_false] at blockMember
  rcases blockMember with rfl | rfl
  · have lower := first54_templateIndex_ge canonicalFirst54Block invocation member
    norm_num only [ringDegree] at productTemplate
    omega
  · exact combination_origin invocation member

/-- The descriptor selects the same concrete recipe as the canonical table. -/
theorem descriptor_template (descriptor : PiRLCProductSchedule.Descriptor) :
    (Data.compactRowTemplates ())[descriptor.compactInvocation.templateIndex]? =
      some (PiRLCCombinationTemplates.template
        (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane) := by
  have selectedIndex : descriptor.compactInvocation.templateIndex =
      PiRLCCombinationTemplates.templateIndex descriptor.source.val descriptor.lane.val := by
    rcases descriptor with ⟨family, source, block, lane, cell⟩
    cases family <;> rfl
  rw [selectedIndex, Data.compactRowTemplates_eq,
    List.getElem?_append_left
      (PiRLCCombinationTemplates.templateIndex_lt descriptor.source.val descriptor.lane)]
  exact PiRLCCombinationTemplates.template_getElem? descriptor.source.val descriptor.lane

/-- Origin is preserved for the actual cached application-shifted records. -/
theorem shifted_origin (context : PerApplicationCachedShift.Context)
    (block : CompactInvocationBlock) (blockMember : block ∈ canonicalCompactBlocks)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand)
    (productTemplate :
      (PerApplicationCachedShift.shiftCompactRowInvocation context invocation).templateIndex <
        2 * ringDegree) :
    ∃ descriptor : PiRLCProductSchedule.Descriptor,
      PerApplicationCachedShift.shiftCompactRowInvocation context invocation =
        PerApplicationCachedShift.shiftCompactRowInvocation context descriptor.compactInvocation := by
  rcases canonical_origin block blockMember invocation member productTemplate with
    ⟨descriptor, rfl⟩
  exact ⟨descriptor, rfl⟩

theorem shifted_localStart (context : PerApplicationCachedShift.Context)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (PerApplicationCachedShift.shiftCompactRowInvocation context
      descriptor.compactInvocation).localStart = descriptor.compactInvocation.localStart := by
  change context.column descriptor.compactInvocation.localStart = _
  unfold PerApplicationCachedShift.Context.column
  apply if_pos
  change descriptor.compactInvocation.localStart < 28784740
  have upper := (PiRLCCombinationScratchGeometry.scratch_contained descriptor).2
  rw [PiRLCCombinationScratchGeometry.scratchEnd_eq] at upper
  omega

/-- Inserting application-private columns does not move the discarded interval. -/
theorem shifted_scratch_contained (context : PerApplicationCachedShift.Context)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    PiRLCCombinationScratchGeometry.scratchStart ≤
        (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).localStart ∧
      (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).localStart +
        PiRLCCombinationScratchGeometry.scratchCount descriptor ≤
          PiRLCCombinationScratchGeometry.scratchEnd := by
  rw [shifted_localStart]
  exact PiRLCCombinationScratchGeometry.scratch_contained descriptor

private theorem shiftedInput_eq_or_ge (context : PerApplicationCachedShift.Context)
    (ranges : List CompactInputRange) (input : Nat) :
    compactInputColumn (ranges.map (PerApplicationCachedShift.shiftCompactInputRange context)) input =
        compactInputColumn ranges input ∨
      Data.physicalLayout.constantColumn ≤ compactInputColumn ranges input ∧
        Data.physicalLayout.constantColumn ≤ compactInputColumn
          (ranges.map (PerApplicationCachedShift.shiftCompactInputRange context)) input := by
  induction ranges with
  | nil => exact Or.inl rfl
  | cons range rest inductionHypothesis =>
      by_cases selected : range.inputStart ≤ input ∧ input < range.inputStart + range.inputCount
      · have baseEq : compactInputColumn (range :: rest) input =
            range.columnStart + (input - range.inputStart) * range.columnStride := by
          simp [compactInputColumn, selected]
        have shiftedEq : compactInputColumn
            ((range :: rest).map (PerApplicationCachedShift.shiftCompactInputRange context)) input =
              context.column range.columnStart + (input - range.inputStart) * range.columnStride := by
          simp [compactInputColumn, PerApplicationCachedShift.shiftCompactInputRange, selected]
        rw [baseEq, shiftedEq]
        by_cases before : range.columnStart < Data.physicalLayout.constantColumn
        · exact Or.inl (by rw [PerApplicationCachedShift.Context.column, if_pos before])
        · right
          rw [PerApplicationCachedShift.Context.column, if_neg before]
          omega
      · have baseEq : compactInputColumn (range :: rest) input =
            compactInputColumn rest input := by
          simp [compactInputColumn, selected]
        have shiftedEq : compactInputColumn
            ((range :: rest).map (PerApplicationCachedShift.shiftCompactInputRange context)) input =
              compactInputColumn
                (rest.map (PerApplicationCachedShift.shiftCompactInputRange context)) input := by
          simp [compactInputColumn, PerApplicationCachedShift.shiftCompactInputRange, selected]
        rw [baseEq, shiftedEq]
        exact inductionHypothesis

private theorem shiftedInput_outside (context : PerApplicationCachedShift.Context)
    (ranges : List CompactInputRange) (input : Nat)
    (outside : compactInputColumn ranges input < PiRLCCombinationScratchGeometry.scratchStart ∨
      PiRLCCombinationScratchGeometry.scratchEnd ≤ compactInputColumn ranges input) :
    compactInputColumn (ranges.map (PerApplicationCachedShift.shiftCompactInputRange context))
        input < PiRLCCombinationScratchGeometry.scratchStart ∨
      PiRLCCombinationScratchGeometry.scratchEnd ≤
        compactInputColumn
          (ranges.map (PerApplicationCachedShift.shiftCompactInputRange context)) input := by
  rcases shiftedInput_eq_or_ge context ranges input with unchanged | after
  · simpa only [unchanged] using outside
  · exact Or.inr (Nat.le_trans (by decide) after.2)

/-- The selected shift cannot move a required input into discarded scratch. -/
theorem shifted_inputs_outside_scratch (context : PerApplicationCachedShift.Context)
    (descriptor : PiRLCProductSchedule.Descriptor) (input : Nat)
    (bounded : input < PiRLCCombinationTemplates.inputCount) :
    compactInputColumn
        (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).inputRanges input <
        PiRLCCombinationScratchGeometry.scratchStart ∨
      PiRLCCombinationScratchGeometry.scratchEnd ≤ compactInputColumn
        (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).inputRanges input := by
  exact shiftedInput_outside context descriptor.compactInvocation.inputRanges input
    (PiRLCCombinationScratchGeometry.inputs_outside_scratch descriptor input bounded)

/-- Shifting public input ranges cannot alias an earlier input with this private output. -/
theorem shifted_output_distinct (context : PerApplicationCachedShift.Context)
    (descriptor : PiRLCProductSchedule.Descriptor) (input : Nat)
    (bounded : input < PiRLCCombinationTemplates.outputInput) :
    compactInputColumn
        (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).inputRanges input ≠
      compactInputColumn
        (PerApplicationCachedShift.shiftCompactRowInvocation context
          descriptor.compactInvocation).inputRanges PiRLCCombinationTemplates.outputInput := by
  have outputBefore : PiRLCCombinationScratchGeometry.inputColumn descriptor 109 <
      Data.physicalLayout.constantColumn :=
    Nat.lt_of_lt_of_le (PiRLCCombinationScratchGeometry.output_before_scratch descriptor)
      (by decide)
  have outputSame : compactInputColumn
      (PerApplicationCachedShift.shiftCompactRowInvocation context
        descriptor.compactInvocation).inputRanges PiRLCCombinationTemplates.outputInput =
      PiRLCCombinationScratchGeometry.inputColumn descriptor 109 := by
    rcases shiftedInput_eq_or_ge context descriptor.compactInvocation.inputRanges 109 with
      unchanged | after
    · exact unchanged
    · exact False.elim (Nat.not_le_of_lt outputBefore after.1)
  rw [outputSame]
  change compactInputColumn
      (descriptor.compactInvocation.inputRanges.map
        (PerApplicationCachedShift.shiftCompactInputRange context)) input ≠
    PiRLCCombinationScratchGeometry.inputColumn descriptor 109
  rcases shiftedInput_eq_or_ge context descriptor.compactInvocation.inputRanges input with
    unchanged | after
  · rw [unchanged]
    exact PiRLCCombinationScratchGeometry.output_distinct descriptor input bounded
  · intro equal
    have later := after.2
    rw [equal] at later
    exact Nat.not_le_of_lt outputBefore later

end NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocationOrigin
