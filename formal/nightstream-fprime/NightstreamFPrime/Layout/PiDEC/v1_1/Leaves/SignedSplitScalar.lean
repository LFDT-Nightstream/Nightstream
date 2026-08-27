import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar

/-!
Owns the exact canonical R1CS footprint of one PiDEC signed scalar split.

The logical circuit remains the authority. This module proves how the fixed
R1CS compiler lowers its Boolean-sign product, sixteen signed-digit products,
and affine radix recomposition. It does not own the 54-coordinate parent,
phase composition, package rows, or assignment generation.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.SignedSplitScalar

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

/-- Stable affine input shape supplied by the PiDEC parent. The digit
non-constant condition excludes a literal coefficient from changing the
canonical product lowering path. -/
structure InputsLinear
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) : Prop where
  parent_mulCount :
    R1CS.mulCount (interface.parent offset) = 0
  digit_mulCount : ∀ index,
    R1CS.mulCount (interface.digit offset index) = 0
  digit_nonconstant : ∀ index,
    Nonconstant (interface.digit offset index)

private theorem difference_nonconstant
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (index : Radix.ChildIndex) :
    Nonconstant
      (interface.digit offset index -
        Lifecycle.PiDEC.v1_1.SignedSplitScalar.signExpr offset) := by
  intro value equality
  change Expr.add _ _ = Expr.const value at equality
  cases equality

private theorem sign_directConstraint_eq_none (offset : Nat) :
    R1CS.directConstraint
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.signConstraint offset) = none := by
  rfl

@[simp] private theorem mulCount_sub (left right : Expr) :
    R1CS.mulCount (left - right) =
      R1CS.mulCount left + R1CS.mulCount right + 1 := by
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) right)) = _
  simp [R1CS.mulCount]
  omega

private theorem digit_directConstraint_eq_none
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (index : Radix.ChildIndex)
    (inputs : InputsLinear interface offset) :
    R1CS.directConstraint
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraint
        interface offset index) = none := by
  have nonAffine : R1CS.lowerAffine
      (interface.digit offset index *
        (interface.digit offset index -
          Lifecycle.PiDEC.v1_1.SignedSplitScalar.signExpr offset)) = none :=
    lowerAffine_mul_eq_none (inputs.digit_nonconstant index)
      (difference_nonconstant interface offset index)
  unfold Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraint
  simp [R1CS.directConstraint, R1CS.affineConstraint, nonAffine]

private theorem sign_freshCount_eq (offset : Nat) :
    R1CS.constraintFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.signConstraint offset) = 2 := by
  unfold R1CS.constraintFreshCount
  rw [sign_directConstraint_eq_none]
  simp [Lifecycle.PiDEC.v1_1.SignedSplitScalar.signConstraint,
    Lifecycle.PiDEC.v1_1.SignedSplitScalar.signBitExpr,
    R1CS.mulCount, mulCount_sub]

private theorem digit_freshCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (index : Radix.ChildIndex)
    (inputs : InputsLinear interface offset) :
    R1CS.constraintFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraint
        interface offset index) = 4 := by
  unfold R1CS.constraintFreshCount
  rw [digit_directConstraint_eq_none interface offset index inputs]
  simp [Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraint,
    Lifecycle.PiDEC.v1_1.SignedSplitScalar.signExpr,
    Lifecycle.PiDEC.v1_1.SignedSplitScalar.signBitExpr,
    R1CS.mulCount, mulCount_sub, inputs.digit_mulCount index]

private theorem weightedFold_affine :
    ∀ (values : List Expr) (weights : List F),
      (∀ value ∈ values, R1CS.IsAffine value) →
      R1CS.IsAffine
        ((values.zip weights).foldr
          (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0)
  | [], _, _ => R1CS.isAffine_const 0
  | _ :: _, [], _ => R1CS.isAffine_const 0
  | value :: values, weight :: weights, affine => by
      apply R1CS.IsAffine.add
      · exact R1CS.IsAffine.const_mul weight (affine value (by simp))
      · apply weightedFold_affine values weights
        intro item member
        exact affine item (by simp [member])

private theorem recomposeExpr_affine
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.IsAffine
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.recomposeExpr
        interface offset) := by
  unfold Lifecycle.PiDEC.v1_1.SignedSplitScalar.recomposeExpr
  apply weightedFold_affine
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact isAffine_of_mulCount_zero _ (inputs.digit_mulCount index)

private theorem recompositionConstraint_affine
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.IsAffine
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.recompositionConstraint
        interface offset) := by
  unfold Lifecycle.PiDEC.v1_1.SignedSplitScalar.recompositionConstraint
  change R1CS.IsAffine
    (.add
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.recomposeExpr interface offset)
      (.mul (.const (-1)) (interface.parent offset)))
  apply R1CS.IsAffine.add
  · exact recomposeExpr_affine interface offset inputs
  · exact R1CS.IsAffine.const_mul (-1)
      (isAffine_of_mulCount_zero _ inputs.parent_mulCount)

private theorem recomposition_freshCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.constraintFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.recompositionConstraint
        interface offset) = 0 :=
  R1CS.constraintFreshCount_eq_zero_of_affine _
    (recompositionConstraint_affine interface offset inputs)

private theorem totalFreshCount_ofFn {count : Nat}
    (constraints : Fin count → Expr) (cost : Fin count → Nat)
    (pointwise : ∀ slot,
      R1CS.constraintFreshCount (constraints slot) = cost slot) :
    R1CS.totalFreshCount (List.ofFn constraints) =
      (List.ofFn cost).sum := by
  unfold R1CS.totalFreshCount
  rw [List.map_ofFn]
  apply congrArg List.sum
  apply congrArg List.ofFn
  funext slot
  exact pointwise slot

private theorem digitConstraints_totalFreshCount
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraints
        interface offset) = 64 := by
  unfold Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraints
  rw [totalFreshCount_ofFn _ (fun _ => 4)
    (fun index => digit_freshCount_eq interface offset index inputs)]
  rfl

/-- Exact fresh intermediate count for one signed scalar split. -/
theorem totalFreshCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.constraints
        interface offset) = 66 := by
  change R1CS.totalFreshCount
    ([Lifecycle.PiDEC.v1_1.SignedSplitScalar.signConstraint offset] ++
      Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraints
        interface offset ++
      [Lifecycle.PiDEC.v1_1.SignedSplitScalar.recompositionConstraint
        interface offset]) = 66
  rw [R1CS.totalFreshCount_append, R1CS.totalFreshCount_append]
  simp only [R1CS.totalFreshCount, List.map_singleton, List.sum_singleton]
  rw [sign_freshCount_eq]
  change 2 + R1CS.totalFreshCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.digitConstraints
        interface offset) +
      R1CS.constraintFreshCount
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.recompositionConstraint
          interface offset) = 66
  rw [digitConstraints_totalFreshCount interface offset inputs,
    recomposition_freshCount_eq interface offset inputs]

/-- Exact physical row count for one signed scalar split. -/
theorem totalRowCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.constraints
        interface offset) = 84 := by
  rw [R1CS.totalRowCount_eq_fresh_add_length,
    totalFreshCount_eq interface offset inputs]
  have lengthEq :
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.constraints
        interface offset).length = 18 := by
    rw [← Lifecycle.PiDEC.v1_1.SignedSplitScalar.flatConstraints_operations]
    exact Lifecycle.PiDEC.v1_1.SignedSplitScalar.flatConstraints_length_eq
      interface offset
  rw [lengthEq]

private theorem circuit_totalFreshCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
        offset)) = 66 := by
  change R1CS.totalFreshCount
    (flatConstraints
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.operations interface offset)) = 66
  rw [Lifecycle.PiDEC.v1_1.SignedSplitScalar.flatConstraints_operations]
  exact totalFreshCount_eq interface offset inputs

private theorem circuit_totalRowCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
        offset)) = 84 := by
  change R1CS.totalRowCount
    (flatConstraints
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.operations interface offset)) = 84
  rw [Lifecycle.PiDEC.v1_1.SignedSplitScalar.flatConstraints_operations]
  exact totalRowCount_eq interface offset inputs

/-- Parent-facing exact footprint. Parents use this bridge without unfolding
the scalar child operations. -/
def footprint
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface) where
  freshColumnCount := fun _ => 66
  physicalRowCount := fun _ => 84
  freshColumnCount_eq := by
    intro offset
    exact circuit_totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    exact circuit_totalRowCount_eq interface offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (inputs : ∀ offset, InputsLinear interface offset)
    (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
        offset)) = 66 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (inputs : ∀ offset, InputsLinear interface offset)
    (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
        offset)) = 84 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Lifecycle.PiDEC.v1_1.SignedSplitScalar.Interface)
    (inputs : ∀ offset, InputsLinear interface offset)
    (offset : Nat) :
    localLength (Circuit.ops
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
        offset) +
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops
          (Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit interface).main
          offset)) = 67 := by
  change localLength
      (Lifecycle.PiDEC.v1_1.SignedSplitScalar.operations interface offset) +
    R1CS.totalFreshCount
      (flatConstraints
        (Lifecycle.PiDEC.v1_1.SignedSplitScalar.operations interface offset)) = 67
  rw [Lifecycle.PiDEC.v1_1.SignedSplitScalar.localLength_eq,
    Lifecycle.PiDEC.v1_1.SignedSplitScalar.flatConstraints_operations,
    totalFreshCount_eq interface offset (inputs offset)]
  rfl

end NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.SignedSplitScalar
