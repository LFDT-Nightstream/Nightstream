import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionLowering
import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionBounds
import NightstreamFPrime.Layout.R1CS.Completeness

/-! Construct the running-transition physical witness at the wide layout's
source offset. The existing logical gadget and lowering theorem are reused. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionLayout

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionInputs
open Spec.Folding.PiCCS.PaperJoint

theorem physical_complete
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env) :
    ∃ completed,
      AgreesOutside env completed phaseOffset 296138 ∧
      R1CS.RowsHold completed (physicalRows logicalWidth publicFits) := by
  let transition := interface logicalWidth publicFits
  let sourceAssumptions := assumptions relation env
  rcases RunningTransition.completeness transition env phaseOffset
      sourceAssumptions specification with
    ⟨logical, logicalAgrees, logicalRows⟩
  have planScope : ∀ expression ∈
      (plan logicalWidth publicFits).constraints,
      expression.VarsBelow (plan logicalWidth publicFits).firstFresh := by
    rw [plan_constraints, plan_firstFresh]
    change ∀ expression ∈ flatConstraints
        (RunningTransition.operations transition phaseOffset),
      expression.VarsBelow
        (phaseOffset + RunningTransition.exactPrivateCount)
    exact RunningTransition.flatConstraints_varsBelow transition phaseOffset env
      sourceAssumptions
  have planLogical : ConstraintsHold logical
      (plan logicalWidth publicFits).constraints := by
    rw [plan_constraints]
    exact logicalRows
  rcases R1CS.LoweringPlan.complete (plan logicalWidth publicFits)
      logical planScope planLogical with
    ⟨completed, loweringAgrees, physicalRowsHold⟩
  have loweringAgreesAtEnd : AgreesOutside logical completed
      (phaseOffset + localLength
        (RunningTransition.operations transition phaseOffset))
      (physicalFreshColumnCount logicalWidth publicFits) := by
    rw [← logicalColumnCount_eq_localLength logicalWidth publicFits]
    change AgreesOutside logical completed
      (plan logicalWidth publicFits).firstFresh
      (plan logicalWidth publicFits).freshColumnCount
    exact loweringAgrees
  refine ⟨completed, ?_, ?_⟩
  · have completeAgrees := logicalAgrees.append loweringAgreesAtEnd
    rw [RunningTransition.localLength_eq,
      physicalFreshColumnCount_eq relation] at completeAgrees
    exact completeAgrees
  · change R1CS.RowsHold completed (plan logicalWidth publicFits).rows
    exact physicalRowsHold

end NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionLayout
