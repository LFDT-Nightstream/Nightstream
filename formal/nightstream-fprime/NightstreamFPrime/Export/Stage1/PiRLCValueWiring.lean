import NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness
import NightstreamFPrime.Export.Stage1.PiRLCProductSchedule

/-!
Owns the PiCCS forms consumed by the direct PiRLC product. Each scheduled
value resolves through the existing PiCCS source classifier, so pilot and
PiCCS coordinates remain authoritative.

This module does not allocate retained coordinates or construct matrix rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCValueWiring

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

theorem valueSource_support
    (descriptor : PiRLCProductSchedule.Descriptor) :
    PiCCSOrdinarySourceSupport.Source
      (descriptor.valueColumn descriptor.lane) := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family with
  | commitment =>
      by_cases first : source.val = 0
      · left; left
        right; right; right; right
        unfold PiCCSOrdinarySourceSupport.InRange
        simp only [PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.commitmentValueSourceStart, first,
          if_pos]
        have blockBound := block.isLt
        have laneBound := lane.isLt
        norm_num [PiCCSInputs.freshCommitmentStart,
          PiCCSInputs.proofInputStart, PiCCSOrdinaryRetainedBlocks.proofInputCount,
          PiCCSOrdinarySourceSupport.proofInputCount, PiCCSInputs.phaseOffset,
          PiCCSInputs.proofInputColumnCount, PiCCSInputs.freshCommitmentWords,
          PiCCSInputs.roundMessageWords, PiCCSInputs.outputEvaluationWords,
          PiRLCCombinationInvocations.sourceCount,
          PiRLCProductSchedule.Family.blockCount,
          ringDegree] at blockBound laneBound ⊢
        omega
      · left; left; left
        unfold PiCCSOrdinarySourceSupport.InRange
        simp only [PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.commitmentValueSourceStart, first,
          if_neg]
        have sourceBound := source.isLt
        have blockBound := block.isLt
        have laneBound := lane.isLt
        norm_num [PiCCSInputs.runningCommitmentStart,
          PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
          PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
          PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
          PiRLCCombinationInvocations.sourceCount,
          PiRLCProductSchedule.Family.blockCount,
          ringDegree] at sourceBound blockBound laneBound ⊢
        omega
  | publicInput =>
      by_cases first : source.val = 0
      · left; left; right; left
        unfold PiCCSOrdinarySourceSupport.InRange
        simp only [PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.publicInputValueSourceStart, first,
          if_pos]
        have blockBound := block.isLt
        have laneBound := lane.isLt
        norm_num [PilotProduction.priorPublicInputStart,
          PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
          PiRLCCombinationInvocations.sourceCount,
          PiRLCProductSchedule.Family.blockCount,
          ringDegree] at blockBound laneBound ⊢
        omega
      · left; left; left
        unfold PiCCSOrdinarySourceSupport.InRange
        simp only [PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.publicInputValueSourceStart, first,
          if_neg]
        have sourceBound := source.isLt
        have blockBound := block.isLt
        have laneBound := lane.isLt
        norm_num [PiCCSInputs.runningPublicStart,
          PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
          PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
          PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
          PiRLCCombinationInvocations.sourceCount,
          PiRLCProductSchedule.Family.blockCount,
          ringDegree] at sourceBound blockBound laneBound ⊢
        omega
  | evalK =>
      left; left; right; right; right; right
      unfold PiCCSOrdinarySourceSupport.InRange
      simp only [PiRLCProductSchedule.Descriptor.valueColumn,
        PiRLCCombinationInvocations.evalKValueSourceStart]
      have sourceBound := source.isLt
      have cellBound := cell.isLt
      have laneBound := lane.isLt
      norm_num [PiCCSInputs.outputEvaluationStart,
        PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSOrdinaryRetainedBlocks.proofInputCount,
        PiCCSOrdinarySourceSupport.proofInputCount, PiCCSInputs.phaseOffset,
        PiCCSInputs.proofInputColumnCount, PiCCSInputs.freshCommitmentWords,
        PiCCSInputs.roundMessageWords, PiCCSInputs.outputEvaluationWords,
        PiRLCCombinationInvocations.sourceCount,
        PiRLCProductSchedule.Family.cellCount,
        ringDegree] at sourceBound cellBound laneBound ⊢
      omega

  | evalA =>
      left; left; right; right; right; right
      unfold PiCCSOrdinarySourceSupport.InRange
      simp only [PiRLCProductSchedule.Descriptor.valueColumn,
        PiRLCCombinationInvocations.evalAValueSourceStart]
      have sourceBound := source.isLt
      have blockBound := block.isLt
      have cellBound := cell.isLt
      have laneBound := lane.isLt
      norm_num [PiCCSInputs.outputEvaluationStart,
        PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSOrdinaryRetainedBlocks.proofInputCount,
        PiCCSOrdinarySourceSupport.proofInputCount, PiCCSInputs.phaseOffset,
        PiCCSInputs.proofInputColumnCount, PiCCSInputs.freshCommitmentWords,
        PiCCSInputs.roundMessageWords, PiCCSInputs.outputEvaluationWords,
        PiRLCCombinationInvocations.sourceCount,
        PiRLCProductSchedule.Family.blockCount,
        PiRLCProductSchedule.Family.cellCount,
        ringDegree] at sourceBound blockBound cellBound laneBound ⊢
      omega

theorem valueSource_beforePhase
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.valueColumn descriptor.lane < PiCCSInputs.phaseOffset := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.valueColumn,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount] at *
  all_goals
    have sourceBound := source.isLt
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have cellBound := cell.isLt
    norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree,
      PiRLCCombinationInvocations.commitmentValueSourceStart,
      PiRLCCombinationInvocations.publicInputValueSourceStart,
      PiRLCCombinationInvocations.evalKValueSourceStart,
      PiRLCCombinationInvocations.evalAValueSourceStart,
      PiCCSInputs.freshCommitmentStart, PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
      PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
      PiCCSInputs.runningGroupWords, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.roundMessageWords,
      PiCCSInputs.phaseOffset, PiCCSInputs.proofInputColumnCount,
      PiCCSInputs.outputEvaluationWords, PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
      at sourceBound blockBound laneBound cellBound ⊢
  all_goals (try split) <;> omega

def located (invocation : Fin PiRLCProductSchedule.invocationCount) :
    PiCCSOrdinaryDirectPlan.Located
      ((PiRLCProductSchedule.descriptor invocation).valueColumn
        (PiRLCProductSchedule.descriptor invocation).lane) :=
  let descriptor := PiRLCProductSchedule.descriptor invocation
  (PiCCSOrdinaryDirectPlan.classifySource
    (descriptor.valueColumn descriptor.lane)).get
      (PiCCSOrdinaryDirectPlan.classifySource_complete
        (valueSource_support descriptor))

def form
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    SparseForm logicalWidth :=
  (located invocation).location.form geometry

theorem location_sourceColumn
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (located invocation).location.sourceColumn =
      (PiRLCProductSchedule.descriptor invocation).valueColumn
        (PiRLCProductSchedule.descriptor invocation).lane :=
  (located invocation).owns

theorem form_eval_eq_decodedEnv
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (form geometry invocation).eval assignment =
      PiCCSAssignmentSoundness.decodedEnv geometry assignment
        (Spartan.sourceToSpartan
          ((PiRLCProductSchedule.descriptor invocation).valueColumn
            (PiRLCProductSchedule.descriptor invocation).lane)) := by
  have read := (PiCCSAssignmentSoundness.decodedEnv_location geometry assignment
    (located invocation).location).symm
  simpa only [form, location_sourceColumn] using read

/-- Canonical construction reads the same base value through the existing
PiCCS owner coordinates. This theorem is constructor evidence only. -/
theorem form_eval_source
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)) :
    (form geometry invocation).eval assignment =
      PiRLCProductPlan.baseEnv program base
        ((PiRLCProductSchedule.descriptor invocation).valueColumn
          (PiRLCProductSchedule.descriptor invocation).lane) := by
  let descriptor := PiRLCProductSchedule.descriptor invocation
  let source := descriptor.valueColumn descriptor.lane
  have support := valueSource_support descriptor
  have bounded := PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount support
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan source, Spartan.sourceToSpartan_lt source bounded⟩
  have direct := PiCCSOrdinaryDirectPlan.sourceMap_form_eval_of_target geometry
    assignment base groupValue products encodes column
      (PiCCSOrdinarySourceSupport.source_target source support)
  rw [form_eval_eq_decodedEnv]
  change SourceCompiler.sourceEnv
      (fun current => ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form current).eval
        assignment) column.val = _
  rw [SourceCompiler.sourceEnv_at, direct]
  rw [RunningTransitionDirectPlan.transitionEnv_of_outside program base source
    bounded (Or.inl (valueSource_beforePhase descriptor))]
  rfl

end NightstreamFPrime.Export.Stage1.PiRLCValueWiring
