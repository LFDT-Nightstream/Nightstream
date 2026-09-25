import NightstreamFPrime.Export.Stage1.Wide.PiRLCSourceOutput
import NightstreamFPrime.Circuit.SequenceValues
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommonSource
import NightstreamFPrime.Export.Stage1.Wide.SamplerSourceValues

/-! Every physical PiRLC prefix output equals the direct retained sum.
The quotient transport uses the same checked challenges and right operands. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCTransportValues

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open ProductionRelation Spec.Folding.PiCCS.PaperJoint
open PiRLCSourceOutput (familyInterface familyStart)

abbrev Family := PiRLCProductSchedule.Family

local instance (family : Family) : NeZero family.cellCount := ⟨by cases family <;> decide⟩

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private abbrev parentInterface := Layout.Stage1.Wide.PiRLCInputs.interface
  (logicalWidth := width) (publicFits := fits)
private abbrev parentStart := Layout.Stage1.Wide.PiRLCInputs.phaseOffset

theorem reference_output_column (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.outputColumn = SourceAssignment.productStart + descriptor.invocation.val := by
  have start : descriptor.family.logicalStart = SourceAssignment.productStart +
      (match descriptor.family with | .commitment => 0 | .publicInput => 20196 | .evalK => 24786 | .evalA => 26622) := by
    cases descriptor.family <;> rfl
  rw [PiRLCProductSchedule.Descriptor.outputColumn, start, PiRLCProductSchedule.Descriptor.invocation_val]
  simp only [PiRLCProductSchedule.Descriptor.logicalIndex, PiRLCCombinationInvocations.stepSize,
    PiRLCCombinationInvocations.logicalIndex, ringDegree, Nat.add_assoc, Nat.mul_assoc]
  rfl

theorem family_output_column (env : Env) (descriptor : PiRLCProductSchedule.Descriptor) :
    CombinationFamily.evalOutputAt env (familyStart descriptor.family) descriptor.source descriptor.block descriptor.cell
        descriptor.lane = env (Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart + descriptor.invocation.val) := by
  change (CombinationStep.output
    (CombinationFamily.stepOffset (familyStart descriptor.family) descriptor.source.val descriptor.family.blockCount
      descriptor.family.cellCount) (CombinationStep.indexOf descriptor.block descriptor.lane descriptor.cell)).eval env = _
  rw [← PiRLCCombinationInvocations.sourceOutput_eq_stepOutput]
  have start : familyStart descriptor.family = Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart +
      (match descriptor.family with | .commitment => 0 | .publicInput => 20196 | .evalK => 24786 | .evalA => 26622) := by
    cases descriptor.family <;> rfl
  change env (familyStart descriptor.family +
    descriptor.source.val * PiRLCCombinationInvocations.stepSize descriptor.family.blockCount descriptor.family.cellCount +
    PiRLCCombinationInvocations.logicalIndex descriptor.family.cellCount descriptor.block.val descriptor.lane.val descriptor.cell.val) = _
  rw [start, PiRLCProductSchedule.Descriptor.invocation_val]
  simp only [PiRLCCombinationInvocations.stepSize, PiRLCCombinationInvocations.logicalIndex,
    ringDegree, Nat.add_assoc, Nat.mul_assoc]
  rfl

theorem source_output (env : Env) (descriptor : PiRLCProductSchedule.Descriptor) :
    SourceAssignment.sourceEnv env descriptor.outputColumn =
      CombinationFamily.evalOutputAt env (familyStart descriptor.family) descriptor.source descriptor.block descriptor.cell
        descriptor.lane := by
  have inside : SourceAssignment.productStart ≤ descriptor.outputColumn ∧
      descriptor.outputColumn < SourceAssignment.productEnd := by
    rw [reference_output_column]
    have bounded : descriptor.invocation.val < 52326 := descriptor.invocation.isLt
    change 19776685 ≤ 19776685 + descriptor.invocation.val ∧ 19776685 + descriptor.invocation.val < 19829011
    omega
  rw [SourceAssignment.sourceEnv, SourceAssignment.source?_product descriptor.outputColumn inside]
  simp only [Option.map_some, Option.getD_some]
  rw [reference_output_column, Nat.add_sub_cancel_left, family_output_column]

theorem family_prefix (relation : ProductionKey.LogicalRelation width fits) (env : Env)
    (assumptions : PiRLC.Wide.Formal.Assumptions relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (physical : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env) (family : Family) :
    CombinationFamily.PrefixHolds (familyInterface (width := width) (fits := fits) family)
      (familyStart family) env := by
  have rows := Layout.PiRLC.Wide.physical_implies_holdsFlat relation
    (parentInterface (width := width) (fits := fits)) parentStart env physical
  rw [PiRLC.Wide.Formal.main_ops] at rows
  let shared := PiRLC.Wide.Formal.atOffset (parentInterface (width := width) (fits := fits)) parentStart
  cases family
  · apply CombinationFamily.childSpecs _ _ env assumptions.commitment
    apply holdsFlat_implies_holds
    exact Sequence.child_rows env _ rows "pirlc.v1_1.commitment_combination"
      (PiRLC.Wide.Formal.commitmentCircuit shared) (PiRLC.Wide.Formal.commitmentOffset parentStart)
      (by exact List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)))
  · apply CombinationFamily.childSpecs _ _ env assumptions.publicInput
    apply holdsFlat_implies_holds
    exact Sequence.child_rows env _ rows "pirlc.v1_1.public_input_combination"
      (PiRLC.Wide.Formal.publicInputCircuit shared) (PiRLC.Wide.Formal.publicInputOffset parentStart)
      (by exact List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _))))
  · apply CombinationFamily.childSpecs _ _ env assumptions.eval_K
    apply holdsFlat_implies_holds
    exact Sequence.child_rows env _ rows "pirlc.v1_1.eval_K_combination"
      (PiRLC.Wide.Formal.evalKCircuit shared) (PiRLC.Wide.Formal.evalKOffset parentStart)
      (by exact List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)))))
  · apply CombinationFamily.childSpecs _ _ env assumptions.eval_A
    apply holdsFlat_implies_holds
    exact Sequence.child_rows env _ rows "pirlc.v1_1.eval_A_combination"
      (PiRLC.Wide.Formal.evalACircuit shared) (PiRLC.Wide.Formal.evalAOffset parentStart)
      (by exact List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _))))))

/-- Each prefix, including all intermediate sources, has the direct sum value. -/
theorem prefix_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (descriptor : PiRLCProductRingSchedule.Descriptor) :
    CombinationFamily.evalOutputAt env (familyStart descriptor.family) descriptor.source descriptor.block descriptor.cell =
      PiRLCValues.output
        (PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)) descriptor.invocation := by
  let base := AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment
  let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program) base
  let values := PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) base
  let inputs := familyInterface (width := width) (fits := fits) descriptor.family
  have phase := Layout.PiRLC.Wide.physical_implies_phaseHolds relation ajtai
    (parentInterface (width := width) (fits := fits)) parentStart env rAssumptions rRows
  have initialEq : List.ofFn initial = PiRLC.Wide.Scalar.evalState env
      ((parentInterface (width := width) (fits := fits)).initialState parentStart) := by
    apply congrArg List.ofFn
    funext lane
    exact PiRLCSourceInputs.initial_value program env application relation ajtai template cAssumptions cRows lane
  have challengeEq (source : Fin 17) : PiRLCValues.challenge initial source.val =
      CombinationFamily.challengeValue inputs (familyStart descriptor.family) env source := by
    funext lane
    rw [PiRLCSourceOutput.challenge_value]
    unfold PiRLCValues.challenge
    rw [initialEq]
    exact congrFun (congrFun phase.response source) lane
  have valueEq (source : Fin 17) : values (PiRLCGeometry.withSource descriptor source).invocation =
      CombinationFamily.inputValue inputs (familyStart descriptor.family) env source descriptor.block descriptor.cell := by
    funext lane
    have copied := PiRLCSourceInputs.input_value program env application
      (PiRLCGeometry.withSource descriptor source).invocation lane
    rw [PiRLCProductRingSchedule.descriptor_laneInvocation, PiRLCProductRingSchedule.descriptor_invocation] at copied
    exact copied.trans (PiRLCSourceOutput.input_value (width := width) (fits := fits) env descriptor.family
      source descriptor.block descriptor.cell lane).symm
  have termEq (source : Fin 17) : PiRLCValues.term initial values descriptor source.val =
      CombinationFamily.term inputs (familyStart descriptor.family) env source descriptor.block descriptor.cell := by
    rw [PiRLCValues.term, dif_pos source.isLt, challengeEq source, valueEq source]
    rfl
  have sums : ∀ count, count ≤ 17 → PiRLCValues.partialSum (PiRLCValues.term initial values descriptor) count =
      CombinationFamily.accumulated inputs (familyStart descriptor.family) env descriptor.block descriptor.cell count := by
    intro count
    induction count with
    | zero => intro _; rfl
    | succ count ih =>
      intro bounded
      have inside : count < 17 := by omega
      have familyInside : count < CombinationFamily.sourceCount := by rwa [CombinationFamily.sourceCount_eq]
      rw [PiRLCValues.partialSum, CombinationFamily.accumulated, dif_pos familyInside,
        ih (by omega), termEq ⟨count, inside⟩]
      rfl
  have output := CombinationFamily.outputAt_eq_accumulatedNat inputs (familyStart descriptor.family) env
    (family_prefix relation env rAssumptions rRows descriptor.family) descriptor.block descriptor.cell
    descriptor.source.val descriptor.source.isLt
  rw [← sums (descriptor.source.val + 1) (by have bound : descriptor.source.val < 17 := descriptor.source.isLt; omega)] at output
  change _ = PiRLCValues.output initial values descriptor.invocation
  rw [PiRLCValues.output, PiRLCProductRingSchedule.descriptor_invocation]
  exact output

theorem valueRing_source (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    AssignmentTransportExecution.valueRing plan (AssignmentTransportSemantics.physicalValues program env application)
      descriptor lane = env (descriptor.valueColumn lane) := by
  obtain ⟨_, _, _, _, _, families, moved, _, _⟩ := AssignmentTransportSemantics.emitted_parts program physicalWidth plan emitted
  let slot := (descriptor.withLane lane).invocation
  have descriptorLane : (descriptor.withLane lane).lane = lane := by cases descriptor; rfl
  have valueColumnEq : (descriptor.withLane lane).valueColumn lane = descriptor.valueColumn lane := by
    simpa only [descriptorLane] using PiRLCProductSchedule.Descriptor.withLane_valueColumn descriptor lane
  have mapped := (AssignmentTransport.moveRuns_source
    (PerApplicationAssignmentTransport.phi81ValueSources program) plan.valueSources moved).2 slot.val
    (by rw [AssignmentTransportSemantics.valueSources_count]; exact slot.isLt)
  dsimp only [slot] at mapped
  rw [PerApplicationAssignmentTransport.phi81ValueSources_at, PiRLCProductSchedule.descriptor_invocation, descriptorLane] at mapped
  let column : Fin (PiRLCProductPlan.baseSourceWidth program) :=
    ⟨(PiRLCProductPlan.valueColumn program (descriptor.withLane lane) lane).val,
      PiRLCProductPlan.valueColumn_val_lt_baseSourceWidth program _ lane⟩
  have read := AssignmentTransportCommonSource.base_value program env application column
    (AffineRuns.sourceAt plan.valueSources slot.val) mapped
  have base : (SourceAssignment.raw program env application).base column =
      PiRLCProductPlan.baseEnv program (SourceAssignment.raw program env application).base
        ((descriptor.withLane lane).valueColumn lane) :=
    (PiRLCProductPlan.baseEnv_valueColumn program (SourceAssignment.raw program env application).base
      (descriptor.withLane lane) lane).symm
  have sourceBound := Layout.Stage1.PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (PiRLCValueWiring.valueSource_support (descriptor.withLane lane))
  have before := PiRLCValueWiring.valueSource_beforePhase (descriptor.withLane lane)
  rw [descriptorLane] at sourceBound before
  have copied := SourceAssignment.raw_source program env application _ sourceBound
  change PiRLCProductPlan.baseEnv program (SourceAssignment.raw program env application).base
    ((descriptor.withLane lane).valueColumn lane) = SourceAssignment.sourceEnv env ((descriptor.withLane lane).valueColumn lane) at copied
  rw [base, copied,
    SourceAssignment.sourceEnv_prefix env _ (lt_of_lt_of_le before (by change 14751804 ≤ 19513117; decide))] at read
  rw [valueColumnEq] at read
  unfold AssignmentTransportExecution.valueRing
  rw [AssignmentTransportSemantics.invocationIndex_eq plan families]
  exact read

theorem valueRing_values (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    AssignmentTransportExecution.valueRing plan (AssignmentTransportSemantics.physicalValues program env application) descriptor =
      PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
        (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
        (PiRLCProductRingSchedule.ofLane descriptor).invocation := by
  funext lane
  rw [valueRing_source program env application physicalWidth plan emitted]
  have copied := PiRLCSourceInputs.input_value program env application
    (PiRLCProductRingSchedule.ofLane descriptor).invocation lane
  rw [PiRLCProductRingSchedule.descriptor_laneInvocation, PiRLCProductRingSchedule.descriptor_invocation,
    PiRLCProductRingSchedule.withLane_ofLane] at copied
  have same : (descriptor.withLane lane).valueColumn lane = descriptor.valueColumn lane := by
    have laneEq : (descriptor.withLane lane).lane = lane := by cases descriptor; rfl
    simpa only [laneEq] using PiRLCProductSchedule.Descriptor.withLane_valueColumn descriptor lane
  rw [same] at copied
  exact copied.symm

/-- The executable virtual suffix is the direct witness's Phi81 quotient. -/
theorem quotient_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (slot : Fin PiRLCProductSchedule.invocationCount) :
    AssignmentTransportExecution.quotientValue plan (AssignmentTransportSemantics.physicalValues program env application) slot =
      PiRLCValues.quotient
        (PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCProductRingSchedule.ringInvocation slot) (PiRLCProductSchedule.descriptor slot).lane := by
  obtain ⟨_, _, _, _, _, _, _, challenges, _⟩ := AssignmentTransportSemantics.emitted_parts program physicalWidth plan emitted
  have initial : Gadgets.Poseidon2.Layer.evalState env
      ((SamplerSourceValues.stageInterface (width := width) (fits := fits)).initialState parentStart) =
      PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
        (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment) := by
    funext lane
    exact (PiRLCSourceInputs.initial_value program env application relation ajtai template cAssumptions cRows lane).symm
  unfold AssignmentTransportExecution.quotientValue
  dsimp only
  rw [SamplerSourceValues.challenge_value program env application plan challenges relation rRows, initial,
    valueRing_values program env application physicalWidth plan emitted]
  simp only [PiRLCValues.quotient, PiRLCProductRingSchedule.ringInvocation,
    PiRLCProductRingSchedule.descriptor_invocation]
  rfl

theorem output_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (block : AssignmentTransport.Values)
    (emitted : AssignmentTransport.commonBlock program .productOutput = .ok block)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (parentInterface (width := width) (fits := fits)) parentStart env)
    (slot : Fin PiRLCProductSchedule.invocationCount) :
    AssignmentTransportSemantics.physicalValues program env application (AffineRuns.sourceAt block.sources slot.val) =
      PiRLCValues.output
        (PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
          (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment))
        (PiRLCProductRingSchedule.ringInvocation slot) (PiRLCProductSchedule.descriptor slot).lane := by
  let descriptor := PiRLCProductSchedule.descriptor slot
  let source := PerApplicationAssignmentBlocks.sourceIndex program .productOutput slot
  have bounded := AssignmentTransport.commonBlock_source_lt program .productOutput block emitted slot
  have mapped := AssignmentTransport.commonBlock_source program .productOutput block emitted slot
  have read := AssignmentTransportCommonSource.base_value program env application ⟨source, bounded⟩ _ mapped
  have base : (SourceAssignment.raw program env application).base ⟨source, bounded⟩ =
      PiRLCProductPlan.baseEnv program (SourceAssignment.raw program env application).base descriptor.outputColumn := by
    exact (PiRLCProductPlan.baseEnv_outputColumn program _ descriptor).symm
  have sourceBound : descriptor.outputColumn < Layout.Stage1.Spartan.SourceColumnCount := by
    rw [reference_output_column, Layout.Stage1.Spartan.sourceColumnCount_eq]
    have inside : descriptor.invocation.val < 52326 := descriptor.invocation.isLt
    change 19776685 + descriptor.invocation.val < 28785018
    omega
  have copied := SourceAssignment.raw_source program env application descriptor.outputColumn sourceBound
  change PiRLCProductPlan.baseEnv program (SourceAssignment.raw program env application).base descriptor.outputColumn =
    SourceAssignment.sourceEnv env descriptor.outputColumn at copied
  rw [base, copied, source_output] at read
  have prefixRead := congrFun (prefix_value program env application relation ajtai template cAssumptions cRows
    rAssumptions rRows (PiRLCProductRingSchedule.ofLane descriptor)) descriptor.lane
  exact read.trans prefixRead

end NightstreamFPrime.Export.Stage1.Wide.PiRLCTransportValues
