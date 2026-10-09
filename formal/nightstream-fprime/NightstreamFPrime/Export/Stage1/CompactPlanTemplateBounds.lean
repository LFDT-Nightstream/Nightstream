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
open PackagePlan

private theorem combination_selection (source : Nat) (lane : Fin ringDegree) :
    (Data.compactRowTemplates ())[
        PiRLCCombinationTemplates.templateIndex source lane.val]? =
      some (PiRLCCombinationTemplates.template (source == 0) lane) := by
  rw [Data.compactRowTemplates_eq]
  exact PiRLCCombinationTemplates.template_getElem? source lane

private theorem combination_family_index_lt (sourceCount : Nat)
    (block : CombinationFamilyBlock) (valueSourceStart : Nat → Nat → Nat → Nat)
    (invocation : CompactRowInvocation)
    (member : invocation ∈ expandCombinationFamily sourceCount block valueSourceStart) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  unfold expandCombinationFamily at member
  rcases List.mem_flatMap.mp member with ⟨source, _sourceMember, indexedMember⟩
  rcases List.mem_ofFn.mp indexedMember with ⟨index, rfl⟩
  let coordinates := NightstreamFPrime.Lifecycle.PiRLC.v1_2.CombinationStep.coordinates index
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

/-- The combination constructor guarantees template bounds for every invocation
they emit. No restriction on their numeric count or geometry fields is needed. -/
theorem block_templateIndex_lt (block : CompactInvocationBlock)
    (invocation : CompactRowInvocation) (member : invocation ∈ block.expand) :
    invocation.templateIndex < (Data.compactRowTemplates ()).length := by
  cases block with
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
