import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition

/-!
Owns canonical R1CS lowering for the reusable PiDEC radix-recomposition
family. All verifier-owned radix weights are constants, so each logical
parent-minus-weighted-children equation is one direct affine row.

This module does not select a commitment or evaluation coordinate family.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.RadixRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.Interface
abbrev circuit :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.circuit
abbrev main :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.main
abbrev operations :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.operations
abbrev constraints :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.constraints
abbrev constraint :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.constraint
abbrev recomposeExpr :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.recomposeExpr

end Logical

/-- Exact affine input shape supplied by a typed PiDEC family. -/
structure InputsLinear {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat) : Prop where
  parent_mulCount : ∀ coordinate,
    R1CS.mulCount (interface.parent offset coordinate) = 0
  child_mulCount : ∀ child coordinate,
    R1CS.mulCount (interface.child offset child coordinate) = 0

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

theorem recomposeExpr_affine {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset)
    (coordinate : Fin coordinateCount) :
    R1CS.IsAffine (Logical.recomposeExpr interface offset coordinate) := by
  unfold Logical.recomposeExpr
  apply weightedFold_affine
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨child, rfl⟩
  exact isAffine_of_mulCount_zero _ (inputs.child_mulCount child coordinate)

theorem constraint_affine {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset)
    (coordinate : Fin coordinateCount) :
    R1CS.IsAffine (Logical.constraint interface offset coordinate) := by
  unfold Logical.constraint
  change R1CS.IsAffine
    (.add (interface.parent offset coordinate)
      (.mul (.const (-1))
        (Logical.recomposeExpr interface offset coordinate)))
  apply R1CS.IsAffine.add
  · exact isAffine_of_mulCount_zero _ (inputs.parent_mulCount coordinate)
  · exact R1CS.IsAffine.const_mul (-1)
      (recomposeExpr_affine interface offset inputs coordinate)

theorem constraint_noFresh {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset)
    (coordinate : Fin coordinateCount) :
    R1CS.constraintFreshCount
      (Logical.constraint interface offset coordinate) = 0 :=
  R1CS.constraintFreshCount_eq_zero_of_affine _
    (constraint_affine interface offset inputs coordinate)

theorem constraint_oneRow {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset)
    (coordinate : Fin coordinateCount) :
    R1CS.constraintRowCount
      (Logical.constraint interface offset coordinate) = 1 :=
  R1CS.constraintRowCount_eq_one_of_affine _
    (constraint_affine interface offset inputs coordinate)

theorem constraints_noFresh {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    ∀ expression ∈ Logical.constraints interface offset,
      R1CS.constraintFreshCount expression = 0 := by
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  exact constraint_noFresh interface offset inputs coordinate

theorem constraints_oneRow {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    ∀ expression ∈ Logical.constraints interface offset,
      R1CS.constraintRowCount expression = 1 := by
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  exact constraint_oneRow interface offset inputs coordinate

theorem totalFreshCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (Logical.constraints interface offset) = 0 :=
  R1CS.totalFreshCount_eq_zero_of_noFresh _
    (constraints_noFresh interface offset inputs)

theorem totalRowCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (Logical.constraints interface offset) =
      coordinateCount := by
  rw [R1CS.totalRowCount_eq_length_of_rowsOne _
    (constraints_oneRow interface offset inputs)]
  change (List.ofFn (Logical.constraint interface offset)).length =
    coordinateCount
  simp

private theorem circuit_totalFreshCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.main interface) offset)) = 0 := by
  change R1CS.totalFreshCount
    (flatConstraints (Logical.operations interface offset)) = 0
  rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.flatConstraints_operations]
  exact totalFreshCount_eq interface offset inputs

private theorem circuit_totalRowCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount) (offset : Nat)
    (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.main interface) offset)) =
      coordinateCount := by
  change R1CS.totalRowCount
    (flatConstraints (Logical.operations interface offset)) = coordinateCount
  rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition.flatConstraints_operations]
  exact totalRowCount_eq interface offset inputs

def footprint {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => coordinateCount
  freshColumnCount_eq := fun offset =>
    circuit_totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    circuit_totalRowCount_eq interface offset (inputs offset)

theorem freshColumnCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount)
    (inputs : ∀ offset, InputsLinear interface offset)
    (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq {coordinateCount : Nat}
    (interface : Logical.Interface coordinateCount)
    (inputs : ∀ offset, InputsLinear interface offset)
    (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) =
      coordinateCount :=
  (footprint interface inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiDEC.v1_1.RadixRecomposition
