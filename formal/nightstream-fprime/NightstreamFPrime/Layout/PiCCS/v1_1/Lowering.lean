import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS steps 1--5.
Obligation: Lower the sole exact PiCCS logical circuit to physical R1CS rows.

Inputs:
- the production logical relation;
- the fixed PiCCS symbolic interface;
- the phase entry offset.

Outputs:
- one ordered logical constraint list;
- one generic R1CS lowering;
- structural logical and physical footprint functions.

Constraint groups:
- P1: all 12 child constraint lists in `Formal.opsAt` order;
- P2: R1CS multiplication rows and terminal zero rows from `R1CS.lowerConstraints`.

Parent coverage:
- `Lifecycle.PiCCS.v1_1.Formal.PhaseHolds` through `Preservation.lean`.

This module does not claim a numeric physical footprint or a domain bound.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The exact constraints of the sole logical PiCCS assembler. -/
def logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Formal.main relation interface) offset)

/-- First column after external inputs and PiCCS-owned logical witnesses. -/
def logicalColumnCount
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (_interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  offset + Formal.privateCount degreeBound

/-- R1CS multiplication columns start after the exact logical interval. -/
def plan
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints relation interface offset
  firstFresh := logicalColumnCount relation interface offset

def lowering
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : R1CS.LoweredConstraints :=
  (plan relation interface offset).lowering

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List R1CS.Row :=
  (plan relation interface offset).rows

def physicalFreshColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).freshColumnCount

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).rowCount

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  (plan relation interface offset).next

theorem logicalConstraints_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (logicalConstraints relation interface offset).length =
      Formal.rowCount degreeBound := by
  unfold logicalConstraints
  exact Formal.flatConstraints_length_eq relation interface offset

theorem logicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    logicalColumnCount relation interface offset =
      offset + localLength (Circuit.ops (Formal.main relation interface) offset) := by
  unfold logicalColumnCount
  rw [Formal.localLength_eq relation interface offset]

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    physicalRowCount relation interface offset =
      R1CS.totalRowCount (logicalConstraints relation interface offset) := by
  change (plan relation interface offset).rowCount =
    R1CS.totalRowCount (plan relation interface offset).constraints
  exact R1CS.LoweringPlan.rowCount_eq _

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    physicalColumnCount relation interface offset =
      logicalColumnCount relation interface offset +
        physicalFreshColumnCount relation interface offset := by
  change (plan relation interface offset).next =
    (plan relation interface offset).firstFresh +
      (plan relation interface offset).freshColumnCount
  exact R1CS.LoweringPlan.next_eq _

theorem logicalConstraints_length_eq_of_degreeBound_eq_nine
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (degreeEq : degreeBound = 9) :
    (logicalConstraints relation interface offset).length = 4581632 := by
  rw [logicalConstraints_length]
  exact Formal.rowCount_eq_of_degreeBound_eq_nine degreeBound degreeEq

theorem logicalColumnCount_eq_of_degreeBound_eq_nine
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (degreeEq : degreeBound = 9) :
    logicalColumnCount relation interface offset = offset + 4581414 := by
  unfold logicalColumnCount
  rw [Formal.privateCount_eq_of_degreeBound_eq_nine degreeBound degreeEq]

end NightstreamFPrime.Layout.PiCCS.v1_1
