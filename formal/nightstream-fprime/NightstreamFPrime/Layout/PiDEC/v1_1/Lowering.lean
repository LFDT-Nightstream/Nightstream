import NightstreamFPrime.Layout.PiDEC.v1_1.Composition

/-!
Owns the one canonical R1CS lowering plan for the exact PiDEC v1_1 phase.
All R1CS intermediates start after the phase's 54 logical sign cells. The
plan lowers the unchanged six-child row order proved by `Composition`.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def logicalColumnCount
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (_interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : Nat :=
  offset + Formal.logicalPrivateCount

def plan
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints relation interface offset
  firstFresh := logicalColumnCount relation interface offset

def lowering
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : R1CS.LoweredConstraints :=
  (plan relation interface offset).lowering

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : List R1CS.Row :=
  (plan relation interface offset).rows

def physicalFreshColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).freshColumnCount

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).rowCount

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).next

@[simp] theorem physicalRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    physicalRows relation interface offset =
      (plan relation interface offset).rows := by
  rfl

theorem physicalRows_eq_lowerConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    physicalRows relation interface offset =
      (R1CS.lowerConstraints (logicalConstraints relation interface offset)
        (logicalColumnCount relation interface offset)).rows := by
  rfl

@[simp] theorem plan_constraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    (plan relation interface offset).constraints =
      logicalConstraints relation interface offset := by
  rfl

@[simp] theorem plan_firstFresh
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    (plan relation interface offset).firstFresh =
      logicalColumnCount relation interface offset := by
  rfl

theorem logicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalColumnCount relation interface offset =
      offset + localLength (Circuit.ops (Formal.main relation interface) offset) := by
  unfold logicalColumnCount
  rw [Formal.localLength_eq]

theorem logicalColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    logicalColumnCount relation interface offset = offset + 54 := by
  unfold logicalColumnCount Formal.logicalPrivateCount
  rfl

theorem physicalFreshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    physicalFreshColumnCount relation interface offset =
      R1CS.totalFreshCount (logicalConstraints relation interface offset) := by
  rfl

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    physicalRowCount relation interface offset =
      R1CS.totalRowCount (logicalConstraints relation interface offset) := by
  exact R1CS.LoweringPlan.rowCount_eq _

theorem physicalRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    (physicalRows relation interface offset).length =
      physicalRowCount relation interface offset := by
  rfl

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) :
    physicalColumnCount relation interface offset =
      logicalColumnCount relation interface offset +
        physicalFreshColumnCount relation interface offset := by
  exact R1CS.LoweringPlan.next_eq _

theorem physicalFreshColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalFreshColumnCount relation interface offset = 3564 := by
  rw [physicalFreshColumnCount_eq]
  exact totalFreshCount_eq relation interface offset inputs

theorem physicalRowCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalRowCount relation interface offset = 7128 := by
  rw [physicalRowCount_eq]
  exact totalRowCount_eq relation interface offset inputs

theorem physicalColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat)
    (inputs : InputShapes relation interface offset) :
    physicalColumnCount relation interface offset = offset + 3618 := by
  rw [physicalColumnCount_eq,
    physicalFreshColumnCount_eq_production relation interface offset inputs,
    logicalColumnCount_eq_production]

end NightstreamFPrime.Layout.PiDEC.v1_1
