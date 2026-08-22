import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, `F`.
Obligation: Share the symbolic sparse-polynomial value
`sum_i gamma^(i-1) f(ct(y'_(i,1)), ..., ct(y'_(i,t)))`.

Inputs:
- the 14 fresh CCS-matrix evaluations for the sole fresh source;
- the relation-owned sparse constraint polynomial.

Outputs:
- the exact symbolic CCS residual consumed by final identity.

Constraint groups:
- none; the reusable sparse evaluator returns a symbolic expression;
- the final-identity leaf constrains that expression.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.ccs_terminal`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits) :
    R1CS.CircuitFootprint (Formal.ccsCircuit relation interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 0
  freshColumnCount_eq := by
    intro offset
    unfold Formal.ccsCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rfl
  physicalRowCount_eq := by
    intro offset
    unfold Formal.ccsCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rfl

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 0 :=
  (footprint relation interface).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 0 :=
  (footprint relation interface).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.ccsCircuit relation interface).main offset)) = 0 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) = 0 := by
    unfold Formal.ccsCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.localLength_eq
        relation (Formal.ccsInterface relation interface) offset
  rw [logicalColumns, freshColumnCount_eq relation interface offset]

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal
