import NightstreamFPrime.Layout.Stage1.RunningTransitionLowering
import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns deterministic physical soundness and constructive completeness for the
Stage 1 running transition. No cryptographic assumption occurs here.
-/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def PhysicalHolds
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows logicalWidth publicFits)

theorem physical_implies_holdsFlat
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env)
    (physical : PhysicalHolds logicalWidth publicFits env) :
    holdsFlat env
      (RunningTransition.operations (interface logicalWidth publicFits)
        phaseOffset) := by
  change R1CS.RowsHold env (plan logicalWidth publicFits).rows at physical
  have logicalRows :=
    R1CS.LoweringPlan.sound (plan logicalWidth publicFits) env physical
  rw [plan_constraints] at logicalRows
  simpa only [logicalConstraints] using logicalRows

theorem physical_implies_specHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (physical : PhysicalHolds logicalWidth publicFits env) :
    RunningTransition.SpecHolds (interface logicalWidth publicFits)
      phaseOffset env := by
  apply RunningTransition.soundness (interface logicalWidth publicFits) env
    phaseOffset (assumptions logicalWidth publicFits relation env)
  exact holdsFlat_implies_holds env _
    (physical_implies_holdsFlat logicalWidth publicFits env physical)

theorem physical_implies_typed_base
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (physical : PhysicalHolds logicalWidth publicFits env)
    (iterationZero : RunningTransition.iterationValue
      (interface logicalWidth publicFits) phaseOffset env = 0) :
    StatementAbsorption.evalRunning
        (outputRunningExpr logicalWidth publicFits) env =
      defaultRunning (logicalWidth := logicalWidth)
        (publicFits := publicFits) :=
  spec_typed_base (physical_implies_specHolds relation env physical)
    iterationZero

theorem physical_implies_typed_recursive
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (physical : PhysicalHolds logicalWidth publicFits env)
    (iterationNonzero : RunningTransition.iterationValue
      (interface logicalWidth publicFits) phaseOffset env ≠ 0) :
    StatementAbsorption.evalRunning
        (outputRunningExpr logicalWidth publicFits) env =
      piDecRunningOutput relation env :=
  spec_typed_recursive_eq_piDecOutput relation
    (physical_implies_specHolds relation env physical) iterationNonzero

theorem physical_complete
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env) :
    ∃ completed,
      AgreesOutside env completed phaseOffset 275386 ∧
      PhysicalHolds logicalWidth publicFits completed := by
  let transition := interface logicalWidth publicFits
  let sourceAssumptions := assumptions logicalWidth publicFits relation env
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

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
