import NightstreamFPrime.Export.Stage1.Wide.CompactScratchGeometry
import NightstreamFPrime.Export.Stage1.StoredDirectPhysicalExecution

/-! The existing stored compact dispatcher after exact wide relocation and
application insertion. Output-only execution preserves rejection and every
coordinate read by the emitted CCS transport. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CompactDirectExecution

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open AssignmentTransportScratch

private theorem inserted_zero (count : Nat) : (ApplicationPackage.insertApplication count).column 0 = .ok 0 := by
  simp only [ApplicationPackage.insertApplication, Layout.Stage1.Wide.SourceOrder.privateColumns_eq,
    show (0 : Nat) < 27859260 by decide, if_true]

private theorem prefix_facts (before after : CompactRowInvocation)
    (emitted : PhysicalRelabel.prefixMap.compact before = .ok after) :
    after.templateIndex = before.templateIndex ∧
      PhysicalRelabel.column before.localStart = .ok after.localStart ∧
      ∀ input, PhysicalRelabel.column (compactInputColumn before.inputRanges input) =
        .ok (compactInputColumn after.inputRanges input) := by
  have zero : PhysicalRelabel.prefixMap.column 0 = .ok 0 := by
    simpa only [PhysicalRelabel.prefixMap] using CompactScratchGeometry.column_zero
  simpa only [PhysicalRelabel.prefixMap] using
    PhysicalCompactRelabel.compact_facts PhysicalRelabel.prefixMap zero before after emitted

private theorem geometry (count : Nat) (descriptor : PiRLCProductSchedule.Descriptor)
    (relocatedRecord actual : CompactRowInvocation)
    (relocated : PhysicalRelabel.prefixMap.compact descriptor.compactInvocation = .ok relocatedRecord)
    (inserted : (ApplicationPackage.insertApplication count).compact relocatedRecord = .ok actual) :
    actual.templateIndex = descriptor.compactInvocation.templateIndex ∧
      (scratchStart ≤ actual.localStart ∧ actual.localStart + PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ scratchEnd) ∧
      (∀ input, input < 110 → Outside (compactInputColumn actual.inputRanges input)) ∧
      (∀ input, input < 109 → compactInputColumn actual.inputRanges input ≠ compactInputColumn actual.inputRanges 109) := by
  obtain ⟨prefixTemplate, prefixLocal, prefixInput⟩ :=
    prefix_facts descriptor.compactInvocation relocatedRecord relocated
  obtain ⟨actualTemplate, actualLocal, actualInput⟩ := PhysicalCompactRelabel.compact_facts
    (ApplicationPackage.insertApplication count) (inserted_zero count) relocatedRecord actual inserted
  have originalContained : PiRLCCombinationScratchGeometry.scratchStart ≤ descriptor.compactInvocation.localStart ∧
      descriptor.compactInvocation.localStart + PiRLCCombinationScratchGeometry.scratchCount descriptor ≤
        PiRLCCombinationScratchGeometry.scratchEnd := PiRLCCombinationScratchGeometry.scratch_contained descriptor
  have contained := CompactScratchGeometry.column_scratch descriptor.compactInvocation.localStart relocatedRecord.localStart
    (PiRLCCombinationScratchGeometry.scratchCount descriptor) prefixLocal originalContained
  have before : relocatedRecord.localStart < Layout.Stage1.Wide.SourceOrder.privateColumns := by
    rw [Layout.Stage1.Wide.SourceOrder.privateColumns_eq]
    rw [scratchEnd_eq] at contained
    omega
  have localEq : actual.localStart = relocatedRecord.localStart := by
    change .ok (if relocatedRecord.localStart < Layout.Stage1.Wide.SourceOrder.privateColumns then relocatedRecord.localStart
      else relocatedRecord.localStart + count) = Except.ok actual.localStart at actualLocal
    rw [if_pos before] at actualLocal
    exact (Except.ok.inj actualLocal).symm
  refine ⟨actualTemplate.trans prefixTemplate, by simpa only [localEq] using contained, ?_, ?_⟩
  · intro input bounded
    have outside := CompactScratchGeometry.column_outside _ _ (prefixInput input)
      (PiRLCCombinationScratchGeometry.inputs_outside_scratch descriptor input bounded)
    have moved := actualInput input
    change .ok (if compactInputColumn relocatedRecord.inputRanges input < Layout.Stage1.Wide.SourceOrder.privateColumns
      then compactInputColumn relocatedRecord.inputRanges input else compactInputColumn relocatedRecord.inputRanges input + count) =
      Except.ok (compactInputColumn actual.inputRanges input) at moved
    rw [← Except.ok.inj moved]
    exact CompactScratchGeometry.inserted_outside count _ outside
  · intro input bounded same
    have left := actualInput input
    have right := actualInput 109
    change .ok (if compactInputColumn relocatedRecord.inputRanges input < Layout.Stage1.Wide.SourceOrder.privateColumns
      then compactInputColumn relocatedRecord.inputRanges input else compactInputColumn relocatedRecord.inputRanges input + count) =
      Except.ok (compactInputColumn actual.inputRanges input) at left
    change .ok (if compactInputColumn relocatedRecord.inputRanges 109 < Layout.Stage1.Wide.SourceOrder.privateColumns
      then compactInputColumn relocatedRecord.inputRanges 109 else compactInputColumn relocatedRecord.inputRanges 109 + count) =
      Except.ok (compactInputColumn actual.inputRanges 109) at right
    have beforeEq := CompactScratchGeometry.inserted_injective count _ _
      ((Except.ok.inj left).trans (same.trans (Except.ok.inj right).symm))
    have inputMap := prefixInput input
    rw [beforeEq] at inputMap
    have original := CompactScratchGeometry.product_injective _ _ _ inputMap (prefixInput 109)
      (CompactScratchGeometry.column_product _ _ (prefixInput 109) (CompactScratchGeometry.output_range descriptor))
    exact PiRLCCombinationScratchGeometry.output_distinct descriptor input bounded original

private theorem selected_template (descriptor : PiRLCProductSchedule.Descriptor) :
    PiRLCCombinationTemplates.templates[descriptor.compactInvocation.templateIndex]? =
      some (PiRLCCombinationTemplates.template
        (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane) := by
  have index : descriptor.compactInvocation.templateIndex =
      PiRLCCombinationTemplates.templateIndex descriptor.source.val descriptor.lane.val := by
    rcases descriptor with ⟨family, source, block, lane, cell⟩
    cases family <;> rfl
  rw [index]
  exact PiRLCCombinationTemplates.template_getElem? descriptor.source.val descriptor.lane

/-- The generic stored compact theorem applies to the exact emitted wide
record. Its premises are only successful metadata relocation and initial
array agreement outside the omitted interval. -/
theorem compact_agree (count : Nat) (descriptor : PiRLCProductSchedule.Descriptor)
    (relocatedRecord actual : CompactRowInvocation)
    (relocated : PhysicalRelabel.prefixMap.compact descriptor.compactInvocation = .ok relocatedRecord)
    (inserted : (ApplicationPackage.insertApplication count).compact relocatedRecord = .ok actual)
    (target : Nat) (left right : Array F) (agree : StoredExecutionSupport.Agree Outside left right) :
    StoredPhysicalExecution.ResultAgree Outside
      (StoredPhysicalExecution.compact PiRLCCombinationTemplates.templates.toArray target actual left)
      (StoredDirectPhysicalExecution.compact PiRLCCombinationTemplates.templates.toArray target actual right) := by
  obtain ⟨index, contained, inputs, distinct⟩ := geometry count descriptor relocatedRecord actual relocated inserted
  let first := PiRLCCombinationInvocations.firstSource descriptor.source.val
  let recipe := PiRLCCombinationTemplates.outputRecipe first descriptor.lane
  let template := PiRLCCombinationTemplates.template first descriptor.lane
  have selected : PiRLCCombinationTemplates.templates.toArray[actual.templateIndex]? = some template := by
    rw [index]
    simpa only [List.getElem?_toArray] using selected_template descriptor
  have scope := PiRLCCompactRecipeScope.combination_outputRecipe first descriptor.lane
  have direct := StoredCompactOutput.compactTemplate_agree Outside
    PiRLCCombinationTemplates.inputCount PiRLCCombinationTemplates.outputInput actual.localStart
    (compactInputColumn actual.inputRanges) recipe left right
    (by change 110 ≤ actual.localStart; rw [scratchStart_eq] at contained; omega)
    (by
      intro input bounded
      have outside := inputs input bounded
      change _ < actual.localStart ∨ actual.localStart + PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ _
      rcases outside with earlier | later
      · exact Or.inl (by omega)
      · exact Or.inr (by omega))
    (by decide) scope distinct
    (by
      apply Expr.VarsSatisfy.mono recipe ((Expr.varsSatisfy_lt_iff_varsBelow recipe 109).mpr scope)
      intro input bounded
      exact inputs input (by omega))
    (by
      intro column outside
      change column < actual.localStart ∨ actual.localStart + PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ column
      rcases outside with earlier | later
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)) agree
  have result := StoredPhysicalExecution.option_result_agree Outside target _ _ direct
  simp only [StoredPhysicalExecution.compact, StoredDirectPhysicalExecution.compact, selected,
    StoredPhysicalExecution.requireWrite, agree.1]
  split_ifs <;> first | exact result | rfl

/-- The native fast branch is selected for every relocated wide product. -/
theorem dispatch_agree (count : Nat) (descriptor : PiRLCProductSchedule.Descriptor)
    (relocatedRecord actual : CompactRowInvocation)
    (relocated : PhysicalRelabel.prefixMap.compact descriptor.compactInvocation = .ok relocatedRecord)
    (inserted : (ApplicationPackage.insertApplication count).compact relocatedRecord = .ok actual)
    (pilot : CircuitPackage) (target : Nat) (left right : Array F)
    (agree : StoredExecutionSupport.Agree Outside left right) :
    StoredPhysicalExecution.ResultAgree Outside
      (StoredPhysicalExecution.executeEvent pilot PiRLCCombinationTemplates.templates.toArray (.compact target actual) left)
      (StoredDirectPhysicalExecution.executeEvent pilot PiRLCCombinationTemplates.templates.toArray (.compact target actual) right) := by
  have selected := (geometry count descriptor relocatedRecord actual relocated inserted).1
  have bounded : actual.templateIndex < 2 * ringDegree := by
    rw [selected]
    rcases descriptor with ⟨family, source, block, lane, cell⟩
    cases family <;> exact PiRLCCombinationTemplates.templateIndex_lt source.val lane
  simp only [StoredPhysicalExecution.executeEvent, StoredDirectPhysicalExecution.executeEvent, if_pos bounded]
  exact compact_agree count descriptor relocatedRecord actual relocated inserted target left right agree

/-- Any successful direct replacement preserves the exact emitted CCS
transport, including its digit checks and public digest. -/
theorem successful_transport_eq (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (count : Nat) (descriptor : PiRLCProductSchedule.Descriptor) (relocatedRecord actual : CompactRowInvocation)
    (relocated : PhysicalRelabel.prefixMap.compact descriptor.compactInvocation = .ok relocatedRecord)
    (inserted : (ApplicationPackage.insertApplication count).compact relocatedRecord = .ok actual)
    (target : Nat) (initial full direct : Array F)
    (fullResult : StoredPhysicalExecution.compact PiRLCCombinationTemplates.templates.toArray target actual initial = .ok full)
    (directResult : StoredDirectPhysicalExecution.compact PiRLCCombinationTemplates.templates.toArray target actual initial = .ok direct) :
    AssignmentTransportExecution.execute program plan physicalWidth (StoredWitnessExecution.asEnv full) =
      AssignmentTransportExecution.execute program plan physicalWidth (StoredWitnessExecution.asEnv direct) := by
  have result := compact_agree count descriptor relocatedRecord actual relocated inserted target initial initial
    ⟨rfl, fun _ _ => rfl⟩
  rw [fullResult, directResult] at result
  exact AssignmentTransportScratch.execute_eq program physicalWidth plan emitted _ _ result.2

end NightstreamFPrime.Export.Stage1.Wide.CompactDirectExecution
