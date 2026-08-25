import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding

/-!
Paper authority: SuperNeo v1.1, Section 7.4, PiRLC input and output.
Obligation: Lower the definitionally shared 17-claim input-binding leaf.

Inputs and outputs:
- the logical `InputBinding.Interface` and its unchanged values.

Constraint groups:
- none; the leaf emits no operation, row, or fresh column.

Parent coverage:
- `PiRLC.Equations.inputFresh`;
- `PiRLC.Equations.sameStructure`;
- `PiRLC.Equations.samePoint`.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.InputBinding

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
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.Interface
        logicalWidth publicFits) :
    R1CS.CircuitFootprint
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.circuit
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
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.circuit
        relation interface).main offset)) = 0 :=
  (footprint relation interface).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.InputBinding
