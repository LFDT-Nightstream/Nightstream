import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding

/-!
Owns the zero-row physical view of the canonical PiRLC output claim.

The output fields are definitionally the final combination variables and the
sampler state. This leaf allocates no witness, fresh R1CS column, copy row, or
boundary row.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.OutputBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.Interface
        logicalWidth publicFits) :
    R1CS.CircuitFootprint
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.circuit
        relation interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 0
  freshColumnCount_eq := by
    intro offset
    rfl
  physicalRowCount_eq := by
    intro offset
    rfl

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.OutputBinding
