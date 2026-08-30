import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptCommonSemantics

/-!
Owns deterministic PiCCS parent semantics in the complete PiRLC sampler
environment. Ordinary rows and transcript specifications are transported by
their exact declared supports. This module adds no row and does not close
PiCCS conformance status.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiCCSCommonPhaseSemantics

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Sampler-complete direct-prefix semantics entail the complete PiCCS parent
specification in the same environment used by PiRLC and PiDEC. -/
theorem semantics_imply_piCcsSpecHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (semantics : DirectPiRLCSamplerCompletePrefixPlan.Semantics relation
      geometry assignment base groupValue products) :
    Lifecycle.PiCCS.v1_1.Formal.SpecHolds relation
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  let transitionEnv := RunningTransitionDirectPlan.transitionEnv application base
  let semanticEnv :=
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
  have ordinaryRows : R1CS.RowsHold semanticEnv
      (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits) := by
    apply R1CS.rowsHold_of_agree
      (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth
        relationPublicFits)
      PiCCSOrdinarySourceSupport.Target transitionEnv semanticEnv
      (PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation)
    · intro column support
      exact PiCCSCommonEnvironmentCustody.semanticEnv_eq_transitionEnv_of_target
        geometry assignment base support
    · exact semantics.prior.piCcsOrdinary
  have packets := PiCCSArithmetic.arithmeticRows_imply_packetHolds
    relationLogicalWidth relationPublicFits semanticEnv ordinaryRows
  have assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear relationLogicalWidth
        relationPublicFits)
      (Spartan.pullback semanticEnv)
  have arithmetic := PiCCSArithmetic.packetHolds_imply_arithmeticSpecs
    relationLogicalWidth relationPublicFits relation semanticEnv assumptions
      packets
  have transcripts := PiCCSTranscriptCommonSemantics.transcriptSpecs_to_common
    relation geometry assignment base groupValue products
      semantics.prior.piCcsTranscript
  refine {
    statementBinding := arithmetic.statementBinding_parent
    statementAbsorption := transcripts.statementAbsorption_parent
    challenge := transcripts.challengeDerivation_parent
    roundTranscript := transcripts.roundTranscript_parent
    initialClaim := arithmetic.initialClaim_parent
    sumcheck := arithmetic.sumcheck_parent
    eval_K := arithmetic.evalK_parent
    eval_A := arithmetic.evalA_parent
    ccs := arithmetic.ccs_parent
    norm := arithmetic.norm_parent
    finalIdentity := arithmetic.finalIdentity_parent
    outputBinding := transcripts.outputBinding_parent relation }

/-- The same sampler-complete semantics entail deterministic PiCCS phase
acceptance for the canonical proof template. -/
theorem semantics_imply_piCcsPhaseHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (template : Proof (ProductionKey.degreeBound relation))
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (semantics : DirectPiRLCSamplerCompletePrefixPlan.Semantics relation
      geometry assignment base groupValue products) :
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))
      template := by
  apply Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds relation ajtai
  exact semantics_imply_piCcsSpecHolds relation geometry assignment base
    groupValue products semantics

end NightstreamFPrime.Export.Stage1.DirectPiCCSCommonPhaseSemantics
