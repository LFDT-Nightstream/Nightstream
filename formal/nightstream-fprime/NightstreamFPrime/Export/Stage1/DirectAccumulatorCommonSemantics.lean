import NightstreamFPrime.Export.Stage1.DirectPiCCSCommonPhaseSemantics
import NightstreamFPrime.Export.Stage1.DirectPiDECCommonPhaseSemantics
import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePhaseSemantics
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds

/-!
Owns deterministic composition of the direct PiCCS, PiRLC, and PiDEC phase
evidence into the one Stage 1 SuperNeo accumulator relation. Every phase uses
the same retained sampler environment. This module adds no row or column and
does not close any phase status.
-/

namespace NightstreamFPrime.Export.Stage1.DirectAccumulatorCommonSemantics

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- One sampler-complete semantic packet forces the exact deterministic
SuperNeo accumulator update. -/
theorem semantics_imply_accumulatorHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
    (vk : KeyDigest)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (encodes : DirectPiRLCSamplerCompletePrefixPlan.Encodes geometry assignment
      base groupValue products)
    (semantics : DirectPiRLCSamplerCompletePrefixPlan.Semantics relation
      geometry assignment base groupValue products)
    (piRlcAssumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        (PiRLCInputs.interface
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits))
        PiRLCInputs.phaseOffset
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))) :
    Lifecycle.Stage1.Accumulator.Holds relation ajtai vk
      (AccumulatorInputs.running relationLogicalWidth relationPublicFits
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)))
      (AccumulatorInputs.fresh relationLogicalWidth relationPublicFits
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)))
      (AccumulatorInputs.proof relation
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)))
      (AccumulatorInputs.output relation
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))) := by
  let commonEnv := Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
  have piCcsPhase :=
    DirectPiCCSCommonPhaseSemantics.semantics_imply_piCcsPhaseHolds relation
      ajtai (AccumulatorInputs.proof relation commonEnv) geometry assignment
      base groupValue products semantics
  have piRlcPhase :=
    DirectPiRLCSamplerCompletePhaseSemantics.semantics_imply_piRlcPhaseHolds
      relation ajtai geometry assignment base groupValue products one encodes
      semantics piRlcAssumptions
  have piDecAssumptions :
      Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        (PiDECArithmetic.phaseInterface relationLogicalWidth
          relationPublicFits)
        PiDECInputs.phaseOffset commonEnv := by
    simpa [PiDECArithmetic.phaseInterface] using
      PiDECInputs.assumptions relation commonEnv
  have piDecPhase :=
    DirectPiDECCommonPhaseSemantics.semantics_imply_piDecPhaseHolds relation
      ajtai geometry assignment base groupValue products semantics
      piDecAssumptions
  apply AccumulatorSemantics.phases_imply_holds relation ajtai vk commonEnv
  · simpa [PiCCSInvocations.parentInterface,
      AccumulatorInputs.piCcsInterface] using piCcsPhase
  · exact piRlcPhase
  · simpa [PiDECArithmetic.phaseInterface] using piDecPhase

end NightstreamFPrime.Export.Stage1.DirectAccumulatorCommonSemantics
