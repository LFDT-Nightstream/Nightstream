import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Owns small proof-only offset facts used to transport deterministic PiCCS
semantics. It adds no row, verifier input, or protocol predicate.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPhaseTransportSupport

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Circuit

/-- Every PiCCS child advances monotonically, so the phase start is at or
before the exact output-binding endpoint. -/
theorem offset_le_finalOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : offset ≤ Formal.finalOffset relation interface offset := by
  unfold Formal.finalOffset Formal.outputBindingOffset
    Formal.finalIdentityOffset Formal.normOffset Formal.ccsOffset
    Formal.evalAOffset Formal.evalKOffset Formal.sumcheckOffset
    Formal.initialClaimOffset Formal.roundTranscriptOffset
    Formal.challengeOffset Formal.statementAbsorptionOffset Formal.nextOffset
  omega

theorem finalOffset_eq_outputBindingEnd
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    Formal.finalOffset relation interface offset =
      Formal.outputBindingOffset relation interface offset + 4076512 := by
  unfold Formal.finalOffset Formal.nextOffset Formal.childLength
    Formal.outputBindingCircuit
  rw [FormalCircuit.withConstantFootprint_main,
    OutputBinding.localLength_eq]

/-- Every outgoing-state expression is supported below the exact PiCCS phase
endpoint. -/
theorem outputBindingFinalState_varsBelow
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Formal.Assumptions relation interface offset env) :
    ∀ lane, (Formal.outputBindingFinalState relation interface offset lane
      ).VarsBelow (Formal.finalOffset relation interface offset) := by
  intro lane
  rw [finalOffset_eq_outputBindingEnd]
  have below := OutputBinding.finalState_varsBelow
    (Formal.outputBindingInterface (Formal.atOffset interface offset))
    (Formal.outputBindingOffset relation interface offset) env
    assumptions.outputBinding lane
  simpa only [OutputBinding.localLength_eq] using below

end NightstreamFPrime.Export.Stage1.PiCCSPhaseTransportSupport
