import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSupport
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptEndpointPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody

/-!
Owns exact environment custody for PiCCS ordinary rows in the complete PiRLC
sampler environment. Every selected source precedes the sampler boundary, so
the sampler view and canonical transition view agree. This module adds no row.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSCommonEnvironmentCustody

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec

private theorem source_beforeSampler {source : Nat}
    (support : PiCCSOrdinarySourceSupport.Source source) :
    source < PiRLCStarts.samplerLogicalStart := by
  rcases support with (external | logicalRange) | fresh
  · rcases external with priorRange | publicRange | outputRange |
      contextRange | proofRange
    · exact Nat.lt_of_lt_of_le priorRange.2 (by
        rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl]
        norm_num [PilotProduction.priorPreimageStart,
          PilotProduction.stateHashWords_eq])
    · exact Nat.lt_of_lt_of_le publicRange.2 (by
        rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl]
        norm_num [PilotProduction.priorPublicInputStart,
          PilotProduction.priorPreimageStart,
          PilotProduction.stateHashWords_eq])
    · exact Nat.lt_of_lt_of_le outputRange.2 (by
        rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl]
        norm_num [PilotProduction.outputPreimageStart,
          PilotProduction.priorPublicInputStart,
          PilotProduction.priorPreimageStart,
          PriorStateHash.publicWidth, PilotProduction.stateHashWords_eq,
          ringDegree, PaperAlgebra.publicRingColumns])
    · exact Nat.lt_of_lt_of_le contextRange.2 (by
        rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl,
          PiCCSInputs.expectedContextStart_eq]
        norm_num [PiCCSInputs.expectedContextWords])
    · exact Nat.lt_of_lt_of_le proofRange.2 (by
        rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl,
          PiCCSInputs.phaseOffset_eq, PiCCSInputs.proofInputStart_eq]
        norm_num)
  · exact Nat.lt_of_lt_of_le logicalRange.2 (by
      rw [show PiRLCStarts.samplerLogicalStart = 19002751 by rfl,
        PiCCSStarts.outputBindingWitnessStart_eq]
      norm_num)
  · simpa [PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, Formal.samplerOffset_eq] using fresh.2

/-- Every final Spartan column used by a PiCCS ordinary row has the canonical
transition value in the complete sampler environment. -/
theorem semanticEnv_eq_transitionEnv_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment :
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment
        F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    {column : Nat} (support : PiCCSOrdinarySourceSupport.Target column) :
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base column =
      RunningTransitionDirectPlan.transitionEnv program base column := by
  rcases support with ⟨source, sourceSupport, mapped⟩
  rw [← mapped]
  exact PiRLCSamplerRetainedCustody.semanticEnv_source_eq_transitionEnv_of_beforeSampler
    geometry assignment base (source_beforeSampler sourceSupport)

/-- Before the sampler starts, the retained PiCCS transcript view and complete
sampler view are the same canonical transition assignment. -/
theorem transcriptEnv_eq_semanticEnv_of_beforeSampler
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth)
    (assignment :
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment
        F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    {column : Nat} (before : column < PiRLCStarts.samplerLogicalStart) :
    Spartan.pullback
        (PiCCSTranscriptEndpointPlan.transcriptEnv program base groupValue
          products) column =
      Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        column := by
  have sourceBound : column < Spartan.SourceColumnCount := by
    apply lt_trans before
    rw [Spartan.sourceColumnCount_eq]
    norm_num [PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      Formal.samplerOffset_eq]
  have mappedBound := Spartan.sourceToSpartan_lt column sourceBound
  unfold Spartan.pullback
  rw [PiCCSTranscriptEndpointPlan.transcriptEnv_eq_transitionEnv_of_lt
    program base groupValue products _ mappedBound]
  rw [PiRLCSamplerRetainedCustody.semanticEnv_source_eq_transitionEnv_of_beforeSampler
    geometry assignment base before]

end NightstreamFPrime.Export.Stage1.PiCCSCommonEnvironmentCustody
