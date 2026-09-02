import NightstreamFPrime.Layout.Stage1.RunningTransitionCost

/-! Owns the sole R1CS lowering plan for the Stage 1 running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def logicalColumnCount : Nat :=
  phaseOffset + RunningTransition.exactPrivateCount

def plan
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : R1CS.LoweringPlan where
  constraints := logicalConstraints logicalWidth publicFits
  firstFresh := logicalColumnCount

def lowering
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.LoweredConstraints :=
  (plan logicalWidth publicFits).lowering

def physicalRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  (plan logicalWidth publicFits).rows

def physicalFreshColumnCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  (plan logicalWidth publicFits).freshColumnCount

def physicalRowCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  (plan logicalWidth publicFits).rowCount

def physicalColumnCount
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  (plan logicalWidth publicFits).next

@[simp] theorem plan_constraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (plan logicalWidth publicFits).constraints =
      logicalConstraints logicalWidth publicFits := by
  rfl

@[simp] theorem plan_firstFresh
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (plan logicalWidth publicFits).firstFresh = logicalColumnCount := by
  rfl

theorem logicalColumnCount_eq : logicalColumnCount = 29040587 := by
  rfl

theorem logicalColumnCount_eq_localLength
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalColumnCount = phaseOffset +
      localLength
        (RunningTransition.operations (interface logicalWidth publicFits)
          phaseOffset) := by
  unfold logicalColumnCount
  rw [RunningTransition.localLength_eq]

@[simp] theorem physicalRows_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    physicalRows logicalWidth publicFits =
      (plan logicalWidth publicFits).rows := by
  rfl

theorem physicalFreshColumnCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalFreshColumnCount logicalWidth publicFits = 296137 := by
  exact totalFreshCount_eq relation

theorem physicalRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount logicalWidth publicFits = 345495 := by
  rw [physicalRowCount]
  exact (R1CS.LoweringPlan.rowCount_eq _).trans (totalRowCount_eq relation)

theorem physicalRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (physicalRows logicalWidth publicFits).length =
      physicalRowCount logicalWidth publicFits := by
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

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
