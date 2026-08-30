import NightstreamFPrime.Export.Stage1.PiRLCSamplerFullSemantics
import NightstreamFPrime.Export.Stage1.PiRLCProductDirectSemantics
import NightstreamFPrime.Export.Stage1.Package

/-!
Owns the phase-local join from the retained PiRLC sampler semantics and exact
compact combination rows to the canonical seven-child PiRLC parent.

This module does not claim package conformance or close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCDirectPhaseSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The retained sampler chain and exact four-family combination packets
assemble the canonical seven-child PiRLC specification. -/
theorem directSampler_imply_specHolds_of_combinationRows
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment)
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (combinationRows : Package.PiRLCCombinationRowsHold
      (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base))
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Formal.SpecHolds relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  let targetEnv :=
    PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base
  let sourceEnv := Spartan.pullback targetEnv
  have chainAssumptions : SamplerChain.Assumptions
      (PiRLCSamplerOrdinaryRows.chainInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCStarts.samplerLogicalStart sourceEnv := by
    simpa [sourceEnv, targetEnv, PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerRows.samplerInterface, PiRLCSamplerRows.sharedInterface] using
        assumptions.sampler
  have samplerChain :=
    PiRLCSamplerFullSemantics.directSemantics_imply_samplerChain relation
      ordinaryGeometry samplerGeometry assignment base groupValue products one
      piCcsEncoding samplerEncoding endpointRows ordinaryRows
      poseidonSemantics sourceHolds chainAssumptions
  refine {
    inputBinding := ?_
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_
    outputBinding := ?_
  }
  · apply InputBinding.soundness
    intro operation member
    cases member
  · simpa [sourceEnv, targetEnv,
      PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerRows.samplerInterface,
      PiRLCSamplerRows.sharedInterface] using samplerChain
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionCommitmentFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.commitmentLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.commitmentFamilyRows_imply_canonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        targetEnv combinationRows.commitment
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionPublicInputFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.publicInputLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.publicInputFamilyRows_imply_canonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        targetEnv combinationRows.publicInput
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionEvalKFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.evalKLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalKFamilyRows_imply_canonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        targetEnv combinationRows.eval_K
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionEvalAFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.evalALogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalAFamilyRows_imply_canonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        targetEnv combinationRows.eval_A
  · apply OutputBinding.soundness
    intro operation member
    cases member

/-- Retained sampler semantics and the self-derived direct product plan
assemble the canonical seven-child PiRLC specification without compact row
acceptance as an authority edge. -/
theorem directSampler_imply_specHolds_of_productSemantics
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment)
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (productSemantics : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0)
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Formal.SpecHolds relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  let targetEnv :=
    PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base
  let sourceEnv := Spartan.pullback targetEnv
  have chainAssumptions : SamplerChain.Assumptions
      (PiRLCSamplerOrdinaryRows.chainInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCStarts.samplerLogicalStart sourceEnv := by
    simpa [sourceEnv, targetEnv, PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerRows.samplerInterface, PiRLCSamplerRows.sharedInterface] using
        assumptions.sampler
  have samplerChain :=
    PiRLCSamplerFullSemantics.directSemantics_imply_samplerChain relation
      ordinaryGeometry samplerGeometry assignment base groupValue products one
      piCcsEncoding samplerEncoding endpointRows ordinaryRows
      poseidonSemantics sourceHolds chainAssumptions
  refine {
    inputBinding := ?_
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_
    outputBinding := ?_
  }
  · apply InputBinding.soundness
    intro operation member
    cases member
  · simpa [sourceEnv, targetEnv,
      PiRLCSamplerOrdinaryRows.chainInterface,
      PiRLCSamplerRows.samplerInterface,
      PiRLCSamplerRows.sharedInterface] using samplerChain
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionCommitmentFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.commitmentLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCProductDirectSemantics.productSemantics_imply_commitmentCanonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        samplerGeometry assignment base productSemantics
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionPublicInputFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.publicInputLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCProductDirectSemantics.productSemantics_imply_publicInputCanonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        samplerGeometry assignment base productSemantics
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionEvalKFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.evalKLogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCProductDirectSemantics.productSemantics_imply_evalKCanonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        samplerGeometry assignment base productSemantics
  · simpa [sourceEnv, targetEnv,
      PiRLCCombinationInvocations.productionEvalAFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      PiRLCStarts.evalALogicalStart,
      PiRLCStarts.phaseLogicalStart] using
      PiRLCProductDirectSemantics.productSemantics_imply_evalACanonical
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        samplerGeometry assignment base productSemantics
  · apply OutputBinding.soundness
    intro operation member
    cases member

/-- The same phase-local evidence entails the deterministic PiRLC accepted
predicate and its verifier-owned sampler replay. -/
theorem directSampler_imply_phaseHolds_of_combinationRows
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment)
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (combinationRows : Package.PiRLCCombinationRowsHold
      (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment base))
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Semantics.PhaseHolds relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  apply Semantics.spec_implies_phaseHolds
  exact directSampler_imply_specHolds_of_combinationRows relation
    ordinaryGeometry samplerGeometry assignment base groupValue products one
    piCcsEncoding samplerEncoding endpointRows ordinaryRows poseidonSemantics
    sourceHolds combinationRows assumptions

/-- The self-derived product plan and retained sampler evidence entail the
deterministic PiRLC accepted predicate and verifier-owned sampler replay. -/
theorem directSampler_imply_phaseHolds_of_productSemantics
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (ordinaryGeometry : PiCCSOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (samplerGeometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiCCSOrdinaryRetainedGeometry.oneColumn ordinaryGeometry) = 1)
    (piCcsEncoding : PiCCSOrdinaryRetainedGeometry.Encodes ordinaryGeometry
      assignment (PiRLCRetainedPreservation.sourceAssignment program base
        groupValue products))
    (samplerEncoding :
      PiRLCSamplerOrdinaryRetainedGeometry.Encodes samplerGeometry assignment
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue
          products))
    (endpointRows : (PiCCSTranscriptEndpointPlan.plan
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
      ordinaryGeometry).RowsZero assignment)
    (ordinaryRows : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv samplerGeometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)))
    (poseidonSemantics :
      PiRLCSamplerPoseidonPreservation.CanonicalSemantics
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry samplerGeometry)
        assignment)
    (sourceHolds : ∀ source,
      PiRLCFirst54DirectPlan.SourceHolds program base source)
    (productSemantics : ∀ invocation,
      (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
        (PiRLCProductPlan.baseEnv program base) = 0)
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base))) :
    Semantics.PhaseHolds relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv samplerGeometry assignment
          base)) := by
  apply Semantics.spec_implies_phaseHolds
  exact directSampler_imply_specHolds_of_productSemantics relation
    ordinaryGeometry samplerGeometry assignment base groupValue products one
    piCcsEncoding samplerEncoding endpointRows ordinaryRows poseidonSemantics
    sourceHolds productSemantics assumptions

end NightstreamFPrime.Export.Stage1.PiRLCDirectPhaseSemantics
