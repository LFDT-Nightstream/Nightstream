import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PiRLCDirectPhaseSemantics

/-!
Owns the phase-local PiRLC lifecycle theorem for the sampler-complete direct
prefix. It composes retained semantic evidence and does not close PiRLC or
package conformance status.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePhaseSemantics

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The sampler-complete direct prefix entails the deterministic PiRLC phase
relation from its own retained semantic evidence. -/
theorem semantics_imply_piRlcPhaseHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
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
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base))) :
    Semantics.PhaseHolds relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  let piDecGeometry :=
    DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry geometry
  let ordinaryGeometry :=
    DirectPiDECPrefixPlan.piCcsOrdinaryGeometry piDecGeometry
  apply PiRLCDirectPhaseSemantics.directSampler_imply_phaseHolds_of_productSemantics
    relation ajtai ordinaryGeometry geometry assignment base groupValue products
  · exact one
  · exact encodes.prior.pilotOrdinary.prior
  · exact encodes.samplerOrdinary
  · simpa [ordinaryGeometry, piDecGeometry,
      DirectPiDECPrefixPlan.piCcsEndpointPlan] using
        semantics.prior.piCcsEndpoint
  · exact semantics.samplerOrdinary
  · simpa [piDecGeometry, DirectPiDECPrefixPlan.poseidonGeometry] using
      semantics.prior.prior.sampler
  · exact semantics.prior.prior.piRlc.first54
  · exact semantics.prior.prior.piRlc.product
  · exact assumptions

/-- Zero rows of the sampler-complete direct prefix, its exact retained
encoding, and the canonical assumptions force deterministic PiRLC acceptance. -/
theorem rowsZero_implies_piRlcPhaseHolds
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
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
    (assumptions : Formal.Assumptions relation
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)))
    (rowsZero : (DirectPiRLCSamplerCompletePrefixPlan.plan relation geometry).RowsZero
      assignment) :
    Semantics.PhaseHolds relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits))
      PiRLCInputs.phaseOffset
      (Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) := by
  have semantics :=
    DirectPiRLCSamplerCompletePrefixPlan.rowsZero_implies_semantics relation
      geometry assignment base groupValue products one encodes rowsZero
  exact semantics_imply_piRlcPhaseHolds relation ajtai geometry assignment base
    groupValue products one encodes semantics assumptions

end NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePhaseSemantics
