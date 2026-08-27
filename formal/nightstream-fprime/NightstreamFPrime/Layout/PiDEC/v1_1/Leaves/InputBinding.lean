import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding

/-!
Owns the zero-row physical view of the exact PiDEC operational input binding.
The binding shares verifier-owned values definitionally and allocates no
witness, R1CS intermediate, copy row, or boundary row.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.InputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.Interface
        logicalWidth publicFits) :
    R1CS.CircuitFootprint
      (NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.circuit
        relation interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 0
  freshColumnCount_eq := by intro offset; rfl
  physicalRowCount_eq := by intro offset; rfl

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.InputBinding
