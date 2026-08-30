import NightstreamFPrime.Export.Stage1.PiCCSCommonEnvironmentCustody

/-!
Owns transport of the four PiCCS transcript specifications into the complete
PiRLC sampler environment. The proof changes no row and no transcript value.
It uses only declared child footprints and the canonical assignment custody
theorem.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSTranscriptCommonSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem statementEnd_beforeSampler
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    PiCCSInvocations.statementWitnessStart +
        (StatementAbsorption.program
          (PiCCSInvocations.statementInterface logicalWidth publicFits)
          PiCCSInvocations.statementWitnessStart).recipes.length <
      PiRLCStarts.samplerLogicalStart := by
  rw [StatementAbsorption.program_recipes_length]
  norm_num [PiCCSInvocations.statementWitnessStart,
    PiCCSStarts.statementWitnessStart_eq, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset_eq]

private theorem challengeEnd_beforeSampler
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    PiCCSInvocations.challengeWitnessStart +
        (ChallengeDerivation.program
          (PiCCSInvocations.challengeInterface logicalWidth publicFits)
          PiCCSInvocations.challengeWitnessStart).recipes.length <
      PiRLCStarts.samplerLogicalStart := by
  rw [ChallengeDerivation.program_recipes_length]
  norm_num [PiCCSInvocations.challengeWitnessStart,
    PiCCSStarts.challengeWitnessStart_eq, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset_eq]

private theorem roundEnd_beforeSampler
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    PiCCSInvocations.roundWitnessStart +
        (RoundTranscript.program
          (PiCCSInvocations.roundInterface logicalWidth publicFits)
          PiCCSInvocations.roundWitnessStart).recipes.length <
      PiRLCStarts.samplerLogicalStart := by
  rw [RoundTranscript.program_recipes_length]
  norm_num [RoundTranscript.perRoundRecipeCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    PiCCSInvocations.roundWitnessStart,
    PiCCSStarts.roundTranscriptWitnessStart_eq,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset_eq]

private theorem outputEnd_beforeSampler
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    PiCCSInvocations.outputWitnessStart +
        localLength (Circuit.ops
          (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.main
            (OutputBinding.duplexInterface
              (PiCCSInvocations.outputInterface logicalWidth publicFits)))
          PiCCSInvocations.outputWitnessStart) <
      PiRLCStarts.samplerLogicalStart := by
  change PiCCSInvocations.outputWitnessStart +
      localLength (Circuit.ops
        (OutputBinding.circuit
          (PiCCSInvocations.outputInterface logicalWidth publicFits)).main
        PiCCSInvocations.outputWitnessStart) <
    PiRLCStarts.samplerLogicalStart
  rw [OutputBinding.localLength_eq]
  norm_num [PiCCSInvocations.outputWitnessStart,
    PiCCSStarts.outputBindingWitnessStart_eq,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset_eq]

/-- The four transcript leaves proved in the retained transcript environment
also hold in the one complete PiRLC sampler environment. -/
theorem transcriptSpecs_to_common
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (assignment :
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment
        F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (specs : PiCCSInvocations.TranscriptSpecs relationLogicalWidth
      relationPublicFits
      (PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
        products)) :
    PiCCSInvocations.TranscriptSpecs relationLogicalWidth relationPublicFits
      (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base) := by
  let transcriptSourceEnv :=
    PiCCSTranscriptEndpointPlan.transcriptEnv application base groupValue
      products
  let semanticSourceEnv :=
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
  let transcriptEnv := Spartan.pullback transcriptSourceEnv
  let commonEnv := Spartan.pullback semanticSourceEnv
  have agrees : ∀ column,
      column < PiRLCStarts.samplerLogicalStart →
        commonEnv column = transcriptEnv column := by
    intro column before
    exact (PiCCSCommonEnvironmentCustody.transcriptEnv_eq_semanticEnv_of_beforeSampler
      geometry assignment base groupValue products before).symm
  have assumptions :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions.production relation
      (PiCCSInvocations.parentInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInputs.phaseOffset
      (PiCCSInputs.externalInputsLinear relationLogicalWidth
        relationPublicFits)
      transcriptEnv
  have statementAssumption : StatementAbsorption.Assumptions
      (PiCCSInvocations.statementInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.statementWitnessStart transcriptEnv := by
    rw [PiCCSInvocations.statementWitnessStart_matches]
    exact assumptions.statementAbsorption
  have challengeAssumption : ChallengeDerivation.Assumptions
      (PiCCSInvocations.challengeInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.challengeWitnessStart transcriptEnv := by
    rw [PiCCSInvocations.challengeWitnessStart_matches]
    exact assumptions.challenge
  have roundAssumption : RoundTranscript.Assumptions
      (PiCCSInvocations.roundInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.roundWitnessStart transcriptEnv := by
    rw [PiCCSInvocations.roundWitnessStart_matches]
    exact assumptions.roundTranscript
  have outputAssumption : OutputBinding.Assumptions
      (PiCCSInvocations.outputInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.outputWitnessStart transcriptEnv := by
    rw [PiCCSInvocations.outputWitnessStart_matches _ _ relation]
    exact assumptions.outputBinding
  refine {
    statementAbsorption := StatementAbsorption.specHolds_of_agree_below
      (PiCCSInvocations.statementInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.statementWitnessStart transcriptEnv commonEnv
      statementAssumption ?_ specs.statementAbsorption
    challengeDerivation := ChallengeDerivation.specHolds_of_agree_below
      (PiCCSInvocations.challengeInterface relationLogicalWidth
        relationPublicFits)
      PiCCSInvocations.challengeWitnessStart transcriptEnv commonEnv
      challengeAssumption ?_ specs.challengeDerivation
    roundTranscript := RoundTranscript.specHolds_of_agree_below
      (PiCCSInvocations.roundInterface relationLogicalWidth relationPublicFits)
      PiCCSInvocations.roundWitnessStart transcriptEnv commonEnv
      roundAssumption ?_ specs.roundTranscript
    outputBinding :=
      NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.specHolds_of_agree_below
        (OutputBinding.duplexInterface
          (PiCCSInvocations.outputInterface relationLogicalWidth
            relationPublicFits))
        PiCCSInvocations.outputWitnessStart transcriptEnv commonEnv
        outputAssumption ?_ specs.outputBinding }
  · intro column before
    exact agrees column (lt_trans before
      (statementEnd_beforeSampler relationLogicalWidth relationPublicFits))
  · intro column before
    exact agrees column (lt_trans before
      (challengeEnd_beforeSampler relationLogicalWidth relationPublicFits))
  · intro column before
    exact agrees column (lt_trans before
      (roundEnd_beforeSampler relationLogicalWidth relationPublicFits))
  · intro column before
    exact agrees column (lt_trans before
      (outputEnd_beforeSampler relationLogicalWidth relationPublicFits))

end NightstreamFPrime.Export.Stage1.PiCCSTranscriptCommonSemantics
