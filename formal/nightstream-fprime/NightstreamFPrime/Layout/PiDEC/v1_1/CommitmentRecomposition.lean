import NightstreamFPrime.Layout.PiDEC.v1_1.RadixRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition

/-!
Owns the typed physical footprint for the 22×54 PiDEC commitment
recomposition family. It selects no evaluation family and adds no copy row.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.CommitmentRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.circuit
abbrev scalarInterface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.scalarInterface
abbrev coordinates :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.coordinates
abbrev coordinateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.coordinateCount
abbrev rowCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.rowCount
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.localLength_eq

end Logical

structure InputsLinear (interface : Logical.Interface) (offset : Nat) : Prop where
  parent_mulCount : ∀ row lane,
    R1CS.mulCount (interface.parent offset row lane) = 0
  child_mulCount : ∀ child row lane,
    R1CS.mulCount (interface.child offset child row lane) = 0

def scalarInputs (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    ∀ offset,
      PiDEC.v1_1.RadixRecomposition.InputsLinear
        (Logical.scalarInterface interface) offset := by
  intro offset
  refine ⟨?_, ?_⟩
  · intro coordinate
    exact inputs offset |>.parent_mulCount
      (Logical.coordinates coordinate).1 (Logical.coordinates coordinate).2
  · intro child coordinate
    exact inputs offset |>.child_mulCount child
      (Logical.coordinates coordinate).1 (Logical.coordinates coordinate).2

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 1188
  freshColumnCount_eq := by
    intro offset
    exact PiDEC.v1_1.RadixRecomposition.freshColumnCount_eq
      (Logical.scalarInterface interface) (scalarInputs interface inputs) offset
  physicalRowCount_eq := by
    intro offset
    calc
      R1CS.totalRowCount
          (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) =
          Logical.coordinateCount :=
        PiDEC.v1_1.RadixRecomposition.physicalRowCount_eq
          (Logical.scalarInterface interface) (scalarInputs interface inputs) offset
      _ = 1188 :=
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition.coordinateCount_eq

theorem freshColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 1188 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 := by
  rw [Logical.localLength_eq, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiDEC.v1_1.CommitmentRecomposition
