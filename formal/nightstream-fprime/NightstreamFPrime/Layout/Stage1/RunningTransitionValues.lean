import NightstreamFPrime.Layout.Stage1.RunningTransitionLowering

/-! Production values for the running-transition endpoints. Core data and
lowering use the shared endpoint and do not import this module. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem logicalColumnCount_eq : logicalColumnCount = 29040587 := by
  rfl

theorem physicalColumnCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount logicalWidth publicFits = 29336724 := by
  rw [physicalColumnCount, R1CS.LoweringPlan.next_eq,
    plan_firstFresh, logicalColumnCount_eq,
    show (plan logicalWidth publicFits).freshColumnCount = 296137 from
      physicalFreshColumnCount_eq relation]

attribute [simp] logicalColumnCount_eq

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
