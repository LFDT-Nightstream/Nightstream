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

/-- Exact syntactic shape of the unmaterialized product exported by this
two-child assembler. -/
structure ProductOutputShape (value : KExpr) : Prop where
  c0_mulCount : R1CS.mulCount value.c0 = 3
  c1_mulCount : R1CS.mulCount value.c1 = 2
  c0_nonAffine : R1CS.lowerAffine value.c0 = none
  c1_nonAffine : R1CS.lowerAffine value.c1 = none

private theorem pointOutput_linear {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) :
    KExprLinear
      (NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointOutput
        interface offset) := by
  unfold NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.pointOutput
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.output
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.program
  apply NightstreamFPrime.Layout.Multilinear.PointEquality.compile_output_linear_of_nonempty
  intro empty
  have lengthZero := congrArg List.length empty
  simp [NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs,
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.canonicalFinIndices_length]
    at lengthZero
  omega

private theorem weightedSum_linear {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (inputs : InputsLinear interface offset)
    (nonempty : interface.coefficients offset ≠ []) :
    KExprLinear
      (NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.weightedSum
        interface offset) := by
  unfold NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.weightedSum
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact compile_output_linear _ _ _ nonempty inputs.coefficient

/-- The exported value is the direct quadratic-extension product of two
materialized child outputs. -/
theorem output_shape {variableCount : Nat}
    (interface : Logical.Interface variableCount) (offset : Nat)
    (positive : 0 < variableCount) (inputs : InputsLinear interface offset)
    (nonempty : interface.coefficients offset ≠ []) :
    ProductOutputShape
      (NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.output
        interface offset) := by
  have pointLinear := pointOutput_linear interface offset positive
  have weightedLinear := weightedSum_linear interface offset inputs nonempty
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.output,
      KExpr.mul, R1CS.mulCount, pointLinear.c0_mulCount,
      pointLinear.c1_mulCount, weightedLinear.c0_mulCount,
      weightedLinear.c1_mulCount]
  · simp [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.output,
      KExpr.mul, R1CS.mulCount, pointLinear.c0_mulCount,
      pointLinear.c1_mulCount, weightedLinear.c0_mulCount,
      weightedLinear.c1_mulCount]
  · simp [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.output,
      KExpr.mul, R1CS.lowerAffine,
      lowerAffine_mul_eq_none pointLinear.c0_nonconstant
        weightedLinear.c0_nonconstant]
  · simp [NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned.output,
      KExpr.mul, R1CS.lowerAffine,
      lowerAffine_mul_eq_none pointLinear.c0_nonconstant
        weightedLinear.c1_nonconstant]

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
