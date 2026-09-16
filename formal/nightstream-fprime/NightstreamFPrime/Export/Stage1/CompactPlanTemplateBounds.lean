import NightstreamFPrime.Export.Stage1.PackagePlan

/-!
Every compact block expansion selects a template in the canonical template
table. The proof uses constructor membership and existing exact template
lookups; it does not enumerate or evaluate the emitted invocation schedule.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CompactPlanTemplateBounds

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Sampling
open PackagePlan

private theorem first54_selection_lt {index : Nat} {template : CompactRowTemplate}
    (selected : PiRLCFirst54Invocations.packageTemplates[index]? = some template) :
    index < (Data.compactRowTemplates ()).length := by
  have found : (Data.compactRowTemplates ())[index]? = some template := by
    simpa only [Data.compactRowTemplates_eq,
      PiRLCFirst54Invocations.packageTemplates] using selected
  rcases List.getElem?_eq_some_iff.mp found with ⟨bounded, _value⟩
  exact bounded

private theorem position_index_lt (source round : Nat)
    (slot : Fin First54Step.slotCount) :
    (PiRLCFirst54Invocations.positionInvocation source round slot.val).templateIndex <
      (Data.compactRowTemplates ()).length := by
  cases round with
  | zero =>
      exact first54_selection_lt
        (PiRLCFirst54Invocations.positionInvocation_zero_template source slot)
  | succ round =>
      exact first54_selection_lt
        (PiRLCFirst54Invocations.positionInvocation_succ_template source round slot)

private theorem value_index_lt (source round : Nat)
    (slot : Fin First54ValueStep.outputCount) :
    (PiRLCFirst54Invocations.valueInvocation source round slot.val).templateIndex <
      (Data.compactRowTemplates ()).length := by
  cases round with
  | zero =>
      exact first54_selection_lt
        (PiRLCFirst54Invocations.valueInvocation_zero_template source slot)
  | succ round =>
      exact first54_selection_lt
        (PiRLCFirst54Invocations.valueInvocation_succ_template source round slot)

private theorem first54_block_index_lt (block : First54InvocationBlock)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  unfold First54InvocationBlock.expand at member
  rcases List.mem_flatMap.mp member with ⟨source, _sourceMember, roundMember⟩
  rcases List.mem_flatMap.mp roundMember with ⟨round, _roundMember, invocationMember⟩
  unfold PiRLCFirst54Invocations.roundInvocations at invocationMember
  rcases List.mem_append.mp invocationMember with positionMember | valueMember
  · unfold PiRLCFirst54Invocations.positionInvocations at positionMember
    rcases List.mem_map.mp positionMember with ⟨slot, _slotMember, rfl⟩
    exact position_index_lt source round slot
  · unfold PiRLCFirst54Invocations.valueInvocations at valueMember
    rcases List.mem_map.mp valueMember with ⟨slot, _slotMember, rfl⟩
    exact value_index_lt source round slot

private theorem combination_selection (source : Nat) (lane : Fin ringDegree) :
    (Data.compactRowTemplates ())[
        PiRLCCombinationTemplates.templateIndex source lane.val]? =
      some (PiRLCCombinationTemplates.template (source == 0) lane) := by
  rw [Data.compactRowTemplates_eq,
    List.getElem?_append_left (PiRLCCombinationTemplates.templateIndex_lt source lane)]
  exact PiRLCCombinationTemplates.template_getElem? source lane

private theorem combination_family_index_lt (sourceCount : Nat)
    (block : CombinationFamilyBlock) (valueSourceStart : Nat → Nat → Nat → Nat)
    (invocation : CompactRowInvocation)
    (member : invocation ∈ expandCombinationFamily sourceCount block valueSourceStart) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  unfold expandCombinationFamily at member
  rcases List.mem_flatMap.mp member with ⟨source, _sourceMember, indexedMember⟩
  rcases List.mem_ofFn.mp indexedMember with ⟨index, rfl⟩
  let coordinates := NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.coordinates index
  change PiRLCCombinationTemplates.templateIndex source coordinates.2.1.val <
    (Data.compactRowTemplates ()).length
  rcases List.getElem?_eq_some_iff.mp (combination_selection source coordinates.2.1) with
    ⟨bounded, _value⟩
  exact bounded

private theorem combination_block_index_lt (block : CombinationInvocationBlock)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  unfold CombinationInvocationBlock.expand at member
  simp only [List.mem_append] at member
  rcases member with ((commitmentMember | publicMember) | evalKMember) | evalAMember
  · exact combination_family_index_lt block.sourceCount block.commitment
      PiRLCCombinationInvocations.commitmentValueSourceStart invocation commitmentMember
  · exact combination_family_index_lt block.sourceCount block.publicInput
      PiRLCCombinationInvocations.publicInputValueSourceStart invocation publicMember
  · exact combination_family_index_lt block.sourceCount block.evalK
      PiRLCCombinationInvocations.evalKValueSourceStart invocation evalKMember
  · exact combination_family_index_lt block.sourceCount block.evalA
      PiRLCCombinationInvocations.evalAValueSourceStart invocation evalAMember

/-- Both block constructors guarantee template bounds for every invocation
they emit. No restriction on their numeric count or geometry fields is needed. -/
theorem block_templateIndex_lt (block : CompactInvocationBlock)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  cases block with
  | first54 block => exact first54_block_index_lt block invocation member
  | combination block => exact combination_block_index_lt block invocation member

/-- The canonical plan inherits the constructor-level bound without expanding
its concrete invocation list. -/
theorem canonical_templateIndex_lt (invocation : CompactRowInvocation)
    (member : invocation ∈ canonicalCompactBlocks.flatMap CompactInvocationBlock.expand) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  rcases List.mem_flatMap.mp member with ⟨block, _blockMember, invocationMember⟩
  exact block_templateIndex_lt block invocation invocationMember

/-- The guarded optional lookup returns the exact proof-indexed template.
This permits a builder to use the bound without a default or chosen template. -/
theorem canonical_template_getElem? (invocation : CompactRowInvocation)
    (member : invocation ∈ canonicalCompactBlocks.flatMap CompactInvocationBlock.expand) :
    (Data.compactRowTemplates ())[invocation.templateIndex]? =
      some ((Data.compactRowTemplates ())[invocation.templateIndex]'
        (canonical_templateIndex_lt invocation member)) := by
  exact List.getElem?_eq_getElem (canonical_templateIndex_lt invocation member)

end NightstreamFPrime.Export.Stage1.CompactPlanTemplateBounds
