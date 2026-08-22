import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS input statement.
Obligation: Share the prior point and separate Eval_K / Eval_A input families.

Inputs:
- the parent PiCCS symbolic interface.

Outputs:
- the exact physical footprint of the parent-facing Statement-binding child.

Constraint groups:
- none; this child shares symbolic expressions and emits no operation.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.statement_binding`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding

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
    (interface : Formal.Interface logicalWidth degreeBound publicFits) :
    R1CS.CircuitFootprint (Formal.statementBindingCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 0
  freshColumnCount_eq := by
    intro offset
    unfold Formal.statementBindingCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rfl
  physicalRowCount_eq := by
    intro offset
    unfold Formal.statementBindingCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rfl

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset)) = 0 :=
  (footprint interface).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset)) = 0 :=
  (footprint interface).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding
