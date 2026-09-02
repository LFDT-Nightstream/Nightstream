import NightstreamFPrime.Export.Stage1.PiRLCProductSourceBlocks
import NightstreamFPrime.Export.Stage1.PiRLCSamplerSelectorCustody

/-!
Owns exact environment custody for the direct PiRLC product schedule.
Schedule-indexed proofs avoid dependent elimination on an arbitrary product
descriptor. This module adds no row and does not close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductSemanticCustody

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem samplerLogicalStart_eq :
    PiRLCStarts.samplerLogicalStart = 20064823 := by
  rfl

private theorem commitmentLogicalStart_eq :
    PiRLCStarts.commitmentLogicalStart = 20328391 := by
  rfl

private theorem phaseFreshStart_eq :
    PiRLCStarts.phaseFreshStart = 20380717 := by
  exact PiRLCStarts.phaseFreshStart_eq

private theorem publicInputLogicalStart_eq :
    PiRLCStarts.publicInputLogicalStart = 20348587 := by
  rfl

private theorem evalKLogicalStart_eq :
    PiRLCStarts.evalKLogicalStart = 20353177 := by
  rfl

private theorem evalALogicalStart_eq :
    PiRLCStarts.evalALogicalStart = 20355013 := by
  rfl

private theorem commitmentValue_beforeSampler
    (index : Fin
      (PiRLCProductSchedule.Family.invocationCount .commitment))
    (lane : Fin ringDegree) :
    (PiRLCProductSchedule.familyDescriptor .commitment index).valueColumn lane <
      PiRLCStarts.samplerLogicalStart := by
  let decoded : Fin PiRLCCombinationInvocations.sourceCount ×
      Fin (PiRLCProductSchedule.Family.privateCount .commitment) :=
    Fin.decodeProd index
  let coordinates := CombinationStep.coordinates decoded.2
  have sourceBound := decoded.1.isLt
  have blockBound := coordinates.1.isLt
  have laneBound := lane.isLt
  change PiRLCCombinationInvocations.commitmentValueSourceStart decoded.1.val
      coordinates.1.val coordinates.2.2.val + lane.val <
    PiRLCStarts.samplerLogicalStart
  rw [samplerLogicalStart_eq]
  unfold PiRLCCombinationInvocations.commitmentValueSourceStart
  change decoded.1.val < 17 at sourceBound
  change coordinates.1.val < 22 at blockBound
  change lane.val < 54 at laneBound
  split
  · norm_num [PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, ringDegree]
    omega
  · norm_num [PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, ringDegree]
    omega

private theorem publicInputValue_beforeSampler
    (index : Fin
      (PiRLCProductSchedule.Family.invocationCount .publicInput))
    (lane : Fin ringDegree) :
    (PiRLCProductSchedule.familyDescriptor .publicInput index).valueColumn lane <
      PiRLCStarts.samplerLogicalStart := by
  let decoded : Fin PiRLCCombinationInvocations.sourceCount ×
      Fin (PiRLCProductSchedule.Family.privateCount .publicInput) :=
    Fin.decodeProd index
  let coordinates := CombinationStep.coordinates decoded.2
  have sourceBound := decoded.1.isLt
  have blockBound := coordinates.1.isLt
  have laneBound := lane.isLt
  change PiRLCCombinationInvocations.publicInputValueSourceStart decoded.1.val
      coordinates.1.val coordinates.2.2.val + lane.val <
    PiRLCStarts.samplerLogicalStart
  rw [samplerLogicalStart_eq]
  unfold PiRLCCombinationInvocations.publicInputValueSourceStart
  change decoded.1.val < 17 at sourceBound
  change coordinates.1.val < 5 at blockBound
  change lane.val < 54 at laneBound
  split
  · norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, ringDegree]
    omega
  · norm_num [PiCCSInputs.runningPublicStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, ringDegree]
    omega

private theorem evalKValue_beforeSampler
    (index : Fin (PiRLCProductSchedule.Family.invocationCount .evalK))
    (lane : Fin ringDegree) :
    (PiRLCProductSchedule.familyDescriptor .evalK index).valueColumn lane <
      PiRLCStarts.samplerLogicalStart := by
  let decoded : Fin PiRLCCombinationInvocations.sourceCount ×
      Fin (PiRLCProductSchedule.Family.privateCount .evalK) :=
    Fin.decodeProd index
  let coordinates := CombinationStep.coordinates decoded.2
  have sourceBound := decoded.1.isLt
  have cellBound := coordinates.2.2.isLt
  have laneBound := lane.isLt
  change PiRLCCombinationInvocations.evalKValueSourceStart decoded.1.val
      coordinates.1.val coordinates.2.2.val + lane.val * 2 <
    PiRLCStarts.samplerLogicalStart
  rw [samplerLogicalStart_eq]
  change decoded.1.val < 17 at sourceBound
  change coordinates.2.2.val < 2 at cellBound
  change lane.val < 54 at laneBound
  norm_num [PiRLCCombinationInvocations.evalKValueSourceStart,
    PiCCSInputs.outputEvaluationStart, PiCCSInputs.roundMessageStart,
    PiCCSInputs.freshCommitmentStart, PiCCSInputs.proofInputStart,
    PiCCSInputs.expectedContextStart, PiCCSInputs.expectedContextWords,
    PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords,
    ringDegree]
  omega

private theorem evalAValue_beforeSampler
    (index : Fin (PiRLCProductSchedule.Family.invocationCount .evalA))
    (lane : Fin ringDegree) :
    (PiRLCProductSchedule.familyDescriptor .evalA index).valueColumn lane <
      PiRLCStarts.samplerLogicalStart := by
  let decoded : Fin PiRLCCombinationInvocations.sourceCount ×
      Fin (PiRLCProductSchedule.Family.privateCount .evalA) :=
    Fin.decodeProd index
  let coordinates := CombinationStep.coordinates decoded.2
  have sourceBound := decoded.1.isLt
  have blockBound := coordinates.1.isLt
  have cellBound := coordinates.2.2.isLt
  have laneBound := lane.isLt
  change PiRLCCombinationInvocations.evalAValueSourceStart decoded.1.val
      coordinates.1.val coordinates.2.2.val + lane.val * 2 <
    PiRLCStarts.samplerLogicalStart
  rw [samplerLogicalStart_eq]
  change decoded.1.val < 17 at sourceBound
  change coordinates.1.val < 14 at blockBound
  change coordinates.2.2.val < 2 at cellBound
  change lane.val < 54 at laneBound
  norm_num [PiRLCCombinationInvocations.evalAValueSourceStart,
    PiCCSInputs.outputEvaluationStart, PiCCSInputs.roundMessageStart,
    PiCCSInputs.freshCommitmentStart, PiCCSInputs.proofInputStart,
    PiCCSInputs.expectedContextStart, PiCCSInputs.expectedContextWords,
    PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords,
    ringDegree]
  omega

private theorem scheduledValue_beforeSampler
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (lane : Fin ringDegree) :
    (PiRLCProductSchedule.descriptor invocation).valueColumn lane <
      PiRLCStarts.samplerLogicalStart := by
  unfold PiRLCProductSchedule.descriptor
  refine Fin.addCases (fun index => ?_) (fun remaining => ?_) invocation
  · simpa using commitmentValue_beforeSampler index lane
  · refine Fin.addCases (fun index => ?_) (fun remaining => ?_) remaining
    · simpa using publicInputValue_beforeSampler index lane
    · refine Fin.addCases (fun index => ?_) (fun index => ?_) remaining
      · simpa using evalKValue_beforeSampler index lane
      · simpa using evalAValue_beforeSampler index lane

private theorem valueColumn_beforeSampler
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    descriptor.valueColumn lane < PiRLCStarts.samplerLogicalStart := by
  have scheduled := scheduledValue_beforeSampler descriptor.invocation lane
  rw [PiRLCProductSchedule.descriptor_invocation] at scheduled
  exact scheduled

private theorem outputColumn_interval
    (descriptor : PiRLCProductSchedule.Descriptor) :
    PiRLCStarts.commitmentLogicalStart ≤ descriptor.outputColumn ∧
      descriptor.outputColumn < PiRLCStarts.phaseFreshStart := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family with
  | commitment =>
      change PiRLCStarts.commitmentLogicalStart ≤
          PiRLCStarts.commitmentLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 22 1 +
            PiRLCCombinationInvocations.logicalIndex 1 block.val lane.val
              cell.val ∧
        PiRLCStarts.commitmentLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 22 1 +
            PiRLCCombinationInvocations.logicalIndex 1 block.val lane.val
              cell.val < PiRLCStarts.phaseFreshStart
      have sourceBound := source.isLt
      have blockBound := block.isLt
      have laneBound := lane.isLt
      have cellBound := cell.isLt
      change source.val < 17 at sourceBound
      change block.val < 22 at blockBound
      change lane.val < 54 at laneBound
      change cell.val < 1 at cellBound
      rw [commitmentLogicalStart_eq, phaseFreshStart_eq]
      norm_num [PiRLCCombinationInvocations.stepSize,
        PiRLCCombinationInvocations.logicalIndex, ringDegree]
      omega
  | publicInput =>
      change PiRLCStarts.commitmentLogicalStart ≤
          PiRLCStarts.publicInputLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 5 1 +
            PiRLCCombinationInvocations.logicalIndex 1 block.val lane.val
              cell.val ∧
        PiRLCStarts.publicInputLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 5 1 +
            PiRLCCombinationInvocations.logicalIndex 1 block.val lane.val
              cell.val < PiRLCStarts.phaseFreshStart
      have sourceBound := source.isLt
      have blockBound := block.isLt
      have laneBound := lane.isLt
      have cellBound := cell.isLt
      change source.val < 17 at sourceBound
      change block.val < 5 at blockBound
      change lane.val < 54 at laneBound
      change cell.val < 1 at cellBound
      rw [commitmentLogicalStart_eq, publicInputLogicalStart_eq,
        phaseFreshStart_eq]
      norm_num [
        PiRLCCombinationInvocations.stepSize,
        PiRLCCombinationInvocations.logicalIndex, ringDegree]
      omega
  | evalK =>
      change PiRLCStarts.commitmentLogicalStart ≤
          PiRLCStarts.evalKLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 1 2 +
            PiRLCCombinationInvocations.logicalIndex 2 block.val lane.val
              cell.val ∧
        PiRLCStarts.evalKLogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 1 2 +
            PiRLCCombinationInvocations.logicalIndex 2 block.val lane.val
              cell.val < PiRLCStarts.phaseFreshStart
      have sourceBound := source.isLt
      have blockBound := block.isLt
      have laneBound := lane.isLt
      have cellBound := cell.isLt
      change source.val < 17 at sourceBound
      change block.val < 1 at blockBound
      change lane.val < 54 at laneBound
      change cell.val < 2 at cellBound
      rw [commitmentLogicalStart_eq, evalKLogicalStart_eq,
        phaseFreshStart_eq]
      norm_num [
        PiRLCCombinationInvocations.stepSize,
        PiRLCCombinationInvocations.logicalIndex, ringDegree]
      omega
  | evalA =>
      change PiRLCStarts.commitmentLogicalStart ≤
          PiRLCStarts.evalALogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 14 2 +
            PiRLCCombinationInvocations.logicalIndex 2 block.val lane.val
              cell.val ∧
        PiRLCStarts.evalALogicalStart + source.val *
              PiRLCCombinationInvocations.stepSize 14 2 +
            PiRLCCombinationInvocations.logicalIndex 2 block.val lane.val
              cell.val < PiRLCStarts.phaseFreshStart
      have sourceBound := source.isLt
      have blockBound := block.isLt
      have laneBound := lane.isLt
      have cellBound := cell.isLt
      change source.val < 17 at sourceBound
      change block.val < 14 at blockBound
      change lane.val < 54 at laneBound
      change cell.val < 2 at cellBound
      rw [commitmentLogicalStart_eq, evalALogicalStart_eq,
        phaseFreshStart_eq]
      norm_num [
        PiRLCCombinationInvocations.stepSize,
        PiRLCCombinationInvocations.logicalIndex, ringDegree]
      omega

private theorem priorColumn_interval
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    PiRLCStarts.commitmentLogicalStart ≤ descriptor.priorColumn ∧
      descriptor.priorColumn < PiRLCStarts.phaseFreshStart := by
  have output := outputColumn_interval
    (descriptor.previousSource notFirst)
  rw [PiRLCProductSchedule.Descriptor.previousSource_outputColumn descriptor
    notFirst] at output
  exact output

private theorem ordinaryLocation_outsideProductInterval
    (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    location.sourceColumn < PiRLCStarts.commitmentLogicalStart ∨
      PiRLCStarts.phaseFreshStart ≤ location.sourceColumn := by
  cases location with
  | poseidon descriptor sourceLane =>
      left
      rcases descriptor with ⟨source, round, lane⟩
      have sourceBound := source.isLt
      have roundBound := round.isLt
      have laneBound := sourceLane.isLt
      change source.val < 17 at sourceBound
      change round.val < 8 at roundBound
      change sourceLane.val < 4 at laneBound
      rw [commitmentLogicalStart_eq]
      cases roundValue : round.val with
      | zero =>
          simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
            PiRLCStarts.samplerSourceLogicalStart]
          rw [samplerLogicalStart_eq]
          omega
      | succ previous =>
          simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
            Sampler.windowOffset, Sampler.windowBase,
            SamplerChain.sourceOffset, DigestWindow.permutationOffset,
            Sampler.logicalPrivateCount, Sampler.entryPrivateCount,
            DigestWindow.logicalPrivateCount, DigestLane.logicalPrivateCount]
          rw [samplerLogicalStart_eq]
          omega
  | logical descriptor position =>
      left
      have sourceBound := descriptor.source.isLt
      have roundBound := descriptor.round.isLt
      have laneBound := descriptor.lane.isLt
      have positionBound := position.isLt
      change descriptor.source.val < 17 at sourceBound
      change descriptor.round.val < 8 at roundBound
      change descriptor.lane.val < 4 at laneBound
      change position.val < 100 at positionBound
      rw [commitmentLogicalStart_eq]
      simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
        PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart]
      rw [samplerLogicalStart_eq]
      omega
  | fresh descriptor position =>
      right
      rw [phaseFreshStart_eq]
      simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryRetainedBlocks.freshSource,
        PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
        PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart]
      rw [phaseFreshStart_eq]
      omega
  | selector source =>
      left
      have sourceBound := source.isLt
      change source.val < 17 at sourceBound
      rw [commitmentLogicalStart_eq]
      simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryDirectSource.selectorSource,
        PiRLCStarts.selectorLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart, First54.positionOffset,
        First54.candidateCount, First54.roundPrivateCount,
        First54Step.slotCount, First54ValueStep.outputCount,
        First54.fullSlot, First54Step.fullSlot]
      rw [samplerLogicalStart_eq]
      omega

private theorem stateLocation_beforeProductInterval
    (location : PiRLCSamplerRetainedCustody.StateLocation) :
    location.sourceColumn < PiRLCStarts.commitmentLogicalStart := by
  have sourceBound := location.source.isLt
  have stepBound := location.step.isLt
  have laneBound := location.lane.isLt
  change location.source.val < 17 at sourceBound
  change location.step.val < 9 at stepBound
  change location.lane.val < 8 at laneBound
  rw [commitmentLogicalStart_eq]
  simp [PiRLCSamplerRetainedCustody.StateLocation.sourceColumn,
    PiRLCSamplerRetainedCustody.stateOutputOffset,
    Sampler.logicalPrivateCount, DigestWindow.logicalPrivateCount]
  rw [samplerLogicalStart_eq]
  omega

private theorem phaseFreshStart_lt_sourceColumnCount :
    PiRLCStarts.phaseFreshStart < Spartan.SourceColumnCount := by
  rw [phaseFreshStart_eq, Spartan.sourceColumnCount_eq]
  norm_num

private theorem ordinaryTarget_none_of_productInterval {column : Nat}
    (lower : PiRLCStarts.commitmentLogicalStart ≤ column)
    (upper : column < PiRLCStarts.phaseFreshStart) :
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan column) = none := by
  unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan column
    (lt_trans upper phaseFreshStart_lt_sourceColumnCount)]
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource column with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      have outside := ordinaryLocation_outsideProductInterval location
      exfalso
      rcases outside with before | after
      · omega
      · omega

private theorem stateTarget_none_of_productInterval {column : Nat}
    (lower : PiRLCStarts.commitmentLogicalStart ≤ column)
    (upper : column < PiRLCStarts.phaseFreshStart) :
    PiRLCSamplerRetainedCustody.classifyStateTarget
        (Spartan.sourceToSpartan column) = none := by
  unfold PiRLCSamplerRetainedCustody.classifyStateTarget
  rw [Spartan.spartanToSource_sourceToSpartan column
    (lt_trans upper phaseFreshStart_lt_sourceColumnCount)]
  cases found : PiRLCSamplerRetainedCustody.classifyStateSource column with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerRetainedCustody.classifyStateSource_sound found
      have before := stateLocation_beforeProductInterval location
      exfalso
      omega

/-- Product recurrence outputs occupy no retained sampler source. The complete
sampler environment therefore uses the canonical transition value. -/
theorem semanticEnv_source_eq_transitionEnv_of_productInterval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    {column : Nat} (lower : PiRLCStarts.commitmentLogicalStart ≤ column)
    (upper : column < PiRLCStarts.phaseFreshStart) :
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
        (Spartan.sourceToSpartan column) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan column) := by
  unfold PiRLCSamplerRetainedCustody.semanticEnv
  rw [ordinaryTarget_none_of_productInterval lower upper,
    stateTarget_none_of_productInterval lower upper]

private theorem samplerLogicalStart_lt_baseConstant :
    PiRLCStarts.samplerLogicalStart <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
      29336446 := by
    exact NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  rw [samplerLogicalStart_eq, constant]
  norm_num

private theorem phaseFreshStart_lt_baseConstant :
    PiRLCStarts.phaseFreshStart <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
      29336446 := by
    exact NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
  rw [phaseFreshStart_eq, constant]
  norm_num

private theorem evalTwo_env_independent (left right : Env) :
    (2 : Expr).eval left = (2 : Expr).eval right := by
  rfl

/-- The product challenge coefficient is the retained final First54 value in
both the complete sampler view and the product-plan base view. -/
theorem semanticEnv_challengeColumn_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        (descriptor.challengeColumn lane) =
      PiRLCProductPlan.baseEnv program base
        (descriptor.challengeColumn lane) := by
  rw [PiRLCProductSourceBlocks.challengeColumn_eq_first54Value]
  simpa [PiRLCFirst54DirectPlan.baseEnv] using
    PiRLCSamplerSelectorCustody.semanticEnv_value_eq_baseEnv geometry
      assignment base
        (PiRLCProductSourceBlocks.challengeValueDescriptor descriptor.source
          lane)

/-- Every product input value precedes the sampler and therefore has the same
canonical transition value in both environments. -/
theorem semanticEnv_valueColumn_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        (descriptor.valueColumn lane) =
      PiRLCProductPlan.baseEnv program base
        (descriptor.valueColumn lane) := by
  have before := valueColumn_beforeSampler descriptor lane
  have privateBound : descriptor.valueColumn lane <
      PiRLCProductPlan.basePackage.layout.constantColumn :=
    lt_trans before samplerLogicalStart_lt_baseConstant
  unfold Spartan.pullback
  rw [PiRLCSamplerRetainedCustody.semanticEnv_source_eq_transitionEnv_of_beforeSampler
    geometry assignment base before]
  exact (PiRLCSamplerRetainedCustody.baseEnv_eq_transitionEnv program base _
    privateBound).symm

/-- Every direct product output lies in the unclaimed product interval and
therefore has the same canonical transition value in both environments. -/
theorem semanticEnv_outputColumn_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        descriptor.outputColumn =
      PiRLCProductPlan.baseEnv program base descriptor.outputColumn := by
  have interval := outputColumn_interval descriptor
  have privateBound : descriptor.outputColumn <
      PiRLCProductPlan.basePackage.layout.constantColumn :=
    lt_trans interval.2 phaseFreshStart_lt_baseConstant
  unfold Spartan.pullback
  rw [semanticEnv_source_eq_transitionEnv_of_productInterval geometry
    assignment base interval.1 interval.2]
  exact (PiRLCSamplerRetainedCustody.baseEnv_eq_transitionEnv program base _
    privateBound).symm

/-- Every nonzero-source prior product output lies in the same unclaimed
product interval in both environments. -/
theorem semanticEnv_priorColumn_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (notFirst : descriptor.source.val ≠ 0) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        descriptor.priorColumn =
      PiRLCProductPlan.baseEnv program base descriptor.priorColumn := by
  have interval := priorColumn_interval descriptor notFirst
  have privateBound : descriptor.priorColumn <
      PiRLCProductPlan.basePackage.layout.constantColumn :=
    lt_trans interval.2 phaseFreshStart_lt_baseConstant
  unfold Spartan.pullback
  rw [semanticEnv_source_eq_transitionEnv_of_productInterval geometry
    assignment base interval.1 interval.2]
  exact (PiRLCSamplerRetainedCustody.baseEnv_eq_transitionEnv program base _
    privateBound).symm

private theorem challengeExpr_evalRing_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    CombinationStep.evalRing
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
        descriptor.challengeExpr =
      CombinationStep.evalRing (PiRLCProductPlan.baseEnv program base)
        descriptor.challengeExpr := by
  funext lane
  simp only [CombinationStep.evalRing,
    PiRLCProductSchedule.Descriptor.challengeExpr, Expr.eval_sub,
    Expr.eval_var]
  rw [semanticEnv_challengeColumn_eq_baseEnv geometry assignment base
    descriptor lane]
  exact congrArg
    (fun value => PiRLCProductPlan.baseEnv program base
      (descriptor.challengeColumn lane) - value)
    (evalTwo_env_independent
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
      (PiRLCProductPlan.baseEnv program base))

private theorem valueExpr_evalRing_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    CombinationStep.evalRing
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
        descriptor.valueExpr =
      CombinationStep.evalRing (PiRLCProductPlan.baseEnv program base)
        descriptor.valueExpr := by
  funext lane
  simpa only [CombinationStep.evalRing,
    PiRLCProductSchedule.Descriptor.valueExpr, Expr.eval_var] using
      semanticEnv_valueColumn_eq_baseEnv geometry assignment base descriptor
        lane

private theorem outputExpr_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.outputExpr.eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      descriptor.outputExpr.eval (PiRLCProductPlan.baseEnv program base) := by
  simpa only [PiRLCProductSchedule.Descriptor.outputExpr, Expr.eval_var] using
    semanticEnv_outputColumn_eq_baseEnv geometry assignment base descriptor

private theorem priorExpr_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.priorExpr.eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      descriptor.priorExpr.eval (PiRLCProductPlan.baseEnv program base) := by
  unfold PiRLCProductSchedule.Descriptor.priorExpr
  split
  · rfl
  · rename_i notFirst
    simpa only [Expr.eval_var] using
      semanticEnv_priorColumn_eq_baseEnv geometry assignment base descriptor
        notFirst

/-- The complete retained sampler environment and the direct product base
environment evaluate every canonical product source constraint identically. -/
theorem sourceConstraint_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.sourceConstraint.eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      descriptor.sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) := by
  rw [PiRLCProductSchedule.Descriptor.sourceConstraint_eq_direct]
  simp only [Expr.eval_sub, Expr.eval_hadd, CombinationStep.mulExpr_eval]
  rw [outputExpr_eval_eq geometry assignment base descriptor,
    priorExpr_eval_eq geometry assignment base descriptor,
    challengeExpr_evalRing_eq geometry assignment base descriptor,
    valueExpr_evalRing_eq geometry assignment base descriptor]

/-- Invocation-indexed direct product semantics yield the exact decoded
source-constraint zero in the complete retained sampler environment. -/
theorem sourceConstraint_zero_of_productSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (product : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    descriptor.sourceConstraint.eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      0 := by
  rw [sourceConstraint_eval_eq geometry assignment base descriptor]
  have zero := product descriptor.invocation
  rw [PiRLCProductSchedule.descriptor_invocation] at zero
  exact zero

end NightstreamFPrime.Export.Stage1.PiRLCProductSemanticCustody
