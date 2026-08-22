import NightstreamFPrime.Layout.Multilinear.PointEquality
import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner

/-!
Owns physical R1CS composition for the reusable owned point-weighted Horner
gadget. The two child row lists are concatenated without a boundary copy row.
This module does not own a protocol coefficient family or point dimension.
-/

namespace NightstreamFPrime.Layout.Multilinear.PointWeightedHorner

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.Interface

abbrev circuit {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.circuit
    interface positive

abbrev pointInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointInterfaceAt
    interface offset

abbrev hornerInterfaceAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.hornerInterfaceAt
    interface offset

abbrev pointCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointCircuitAt
    interface offset

abbrev hornerCircuitAt {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.hornerCircuitAt
    interface offset

abbrev hornerOffset {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat) :=
  NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.hornerOffset
    interface offset

end Logical

/-- Stable physical wire shape for both opaque children. -/
structure InputsLinear {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat) : Prop where
  left : ∀ coordinate,
    KExprLinear (interface.left offset coordinate)
  right : ∀ coordinate,
    KExprLinear (interface.right offset coordinate)
  hornerPoint : KExprLinear (interface.hornerPoint offset)
  coefficient : ∀ coefficient ∈ interface.coefficients offset,
    KExprLinear coefficient

private theorem point_totalFreshCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Logical.pointCircuitAt interface offset).main offset)) =
      24 * variableCount - 7 := by
  unfold Logical.pointCircuitAt
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointCircuitAt
  exact
    NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalFreshCount_of_positive
      (Logical.pointInterfaceAt interface offset) offset positive
      (fun coordinate => ⟨inputs.left coordinate, inputs.right coordinate⟩)

private theorem point_totalRowCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Logical.pointCircuitAt interface offset).main offset)) =
      28 * variableCount - 9 := by
  unfold Logical.pointCircuitAt
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointCircuitAt
  exact
    NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalRowCount_of_positive
      (Logical.pointInterfaceAt interface offset) offset positive
      (fun coordinate => ⟨inputs.left coordinate, inputs.right coordinate⟩)

private theorem horner_totalFreshCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Logical.hornerCircuitAt interface offset).main
        (Logical.hornerOffset interface offset))) =
      7 * ((interface.coefficients offset).length - 1) := by
  unfold Logical.hornerCircuitAt
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.hornerCircuitAt
  apply ownedCircuit_totalFreshCount
  · exact inputs.hornerPoint
  · exact inputs.coefficient

private theorem horner_totalRowCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Logical.hornerCircuitAt interface offset).main
        (Logical.hornerOffset interface offset))) =
      9 * ((interface.coefficients offset).length - 1) := by
  unfold Logical.hornerCircuitAt
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.hornerCircuitAt
  apply ownedCircuit_totalRowCount
  · exact inputs.hornerPoint
  · exact inputs.coefficient

theorem totalFreshCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints
      (Circuit.ops (Logical.circuit interface positive).main offset)) =
      (24 * variableCount - 7) +
        7 * ((interface.coefficients offset).length - 1) := by
  unfold Logical.circuit
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.circuit
  rw [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.main_ops,
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.flatConstraints_opsAt,
    R1CS.totalFreshCount_append,
    point_totalFreshCount interface offset positive inputs,
    horner_totalFreshCount interface offset inputs]

theorem totalRowCount {variableCount : Nat}
    (interface : Logical.Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints
      (Circuit.ops (Logical.circuit interface positive).main offset)) =
      (28 * variableCount - 9) +
        9 * ((interface.coefficients offset).length - 1) := by
  unfold Logical.circuit
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.circuit
  rw [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.main_ops,
    NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.flatConstraints_opsAt,
    R1CS.totalRowCount_append,
    point_totalRowCount interface offset positive inputs,
    horner_totalRowCount interface offset inputs]

/-- Parent-facing exact footprint for the two-child assembler. -/
def footprint {variableCount : Nat}
    (interface : Logical.Interface variableCount) (positive : 0 < variableCount)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface positive) where
  freshColumnCount := fun offset =>
    (24 * variableCount - 7) +
      7 * ((interface.coefficients offset).length - 1)
  physicalRowCount := fun offset =>
    (28 * variableCount - 9) +
      9 * ((interface.coefficients offset).length - 1)
  freshColumnCount_eq := fun offset =>
    totalFreshCount interface positive offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount interface positive offset (inputs offset)

end NightstreamFPrime.Layout.Multilinear.PointWeightedHorner
