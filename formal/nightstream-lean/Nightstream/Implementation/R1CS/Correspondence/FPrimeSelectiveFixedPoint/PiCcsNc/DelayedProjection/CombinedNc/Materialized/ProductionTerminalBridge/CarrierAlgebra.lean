import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.MixedPolynomial

/-!
Sparse-linear and concrete-carrier algebra used by the production terminal
bridge.  This leaf contains no assignment authority or terminal acceptance.

Owns: exact field-algebra identities for the combined carrier terminal expression.
Does not own: production column decoding, row satisfaction, transcript sampling, or child authority.
Emits constraints: none.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.terminal.carrier_algebra` | Reduce the combined carrier terminal expression to its typed field identity. | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

theorem mappedFieldOne :
    ProductionMessageAcceptance.toConcreteField
        (1 : ProjectionProgram.F) = (1 : F) := by
  apply Fin.ext
  rfl

theorem mappedLinear1 (assignment : Nat -> Nat)
    (column coefficient : Nat) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (lcEval assignment [(column, coefficient)])) =
      ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue coefficient) *
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.baseAt assignment column) := by
  have source : ProjectionProgram.residue
      (lcEval assignment [(column, coefficient)]) =
      ProjectionProgram.residue coefficient *
        ProjectionProgram.baseAt assignment column := by
    apply Fin.ext
    simp [ProjectionProgram.baseAt, ProjectionProgram.residue, lcEval,
      Fin.val_add, Fin.val_mul]
  rw [source, ProductionMessageAcceptance.toConcreteField_mul]

theorem mappedLinear2 (assignment : Nat -> Nat)
    (first firstCoefficient second secondCoefficient : Nat) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (lcEval assignment
          [(first, firstCoefficient), (second, secondCoefficient)])) =
      ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue firstCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment first) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue secondCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment second) := by
  have source : ProjectionProgram.residue (lcEval assignment
      [(first, firstCoefficient), (second, secondCoefficient)]) =
      ProjectionProgram.residue firstCoefficient *
          ProjectionProgram.baseAt assignment first +
        ProjectionProgram.residue secondCoefficient *
          ProjectionProgram.baseAt assignment second := by
    apply Fin.ext
    simp [ProjectionProgram.baseAt, ProjectionProgram.residue, lcEval,
      Fin.val_add, Fin.val_mul]
  rw [source, ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul]

theorem mappedLinear3 (assignment : Nat -> Nat)
    (first firstCoefficient second secondCoefficient third thirdCoefficient : Nat) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (lcEval assignment
          [(first, firstCoefficient), (second, secondCoefficient),
            (third, thirdCoefficient)])) =
      (ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue firstCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment first) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue secondCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment second)) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue thirdCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment third) := by
  have source : ProjectionProgram.residue (lcEval assignment
      [(first, firstCoefficient), (second, secondCoefficient),
        (third, thirdCoefficient)]) =
      (ProjectionProgram.residue firstCoefficient *
          ProjectionProgram.baseAt assignment first +
        ProjectionProgram.residue secondCoefficient *
          ProjectionProgram.baseAt assignment second) +
        ProjectionProgram.residue thirdCoefficient *
          ProjectionProgram.baseAt assignment third := by
    apply Fin.ext
    simp [ProjectionProgram.baseAt, ProjectionProgram.residue, lcEval,
      Fin.val_add, Fin.val_mul]
  rw [source, ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul]

theorem mappedLinear4 (assignment : Nat -> Nat)
    (first firstCoefficient second secondCoefficient third thirdCoefficient
      fourth fourthCoefficient : Nat) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (lcEval assignment
          [(first, firstCoefficient), (second, secondCoefficient),
            (third, thirdCoefficient), (fourth, fourthCoefficient)])) =
      ((ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue firstCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment first) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue secondCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment second)) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue thirdCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment third)) +
        ProductionMessageAcceptance.toConcreteField
          (ProjectionProgram.residue fourthCoefficient) *
          ProductionMessageAcceptance.toConcreteField
            (ProjectionProgram.baseAt assignment fourth) := by
  have source : ProjectionProgram.residue (lcEval assignment
      [(first, firstCoefficient), (second, secondCoefficient),
        (third, thirdCoefficient), (fourth, fourthCoefficient)]) =
      ((ProjectionProgram.residue firstCoefficient *
          ProjectionProgram.baseAt assignment first +
        ProjectionProgram.residue secondCoefficient *
          ProjectionProgram.baseAt assignment second) +
        ProjectionProgram.residue thirdCoefficient *
          ProjectionProgram.baseAt assignment third) +
        ProjectionProgram.residue fourthCoefficient *
          ProjectionProgram.baseAt assignment fourth := by
    apply Fin.ext
    simp [ProjectionProgram.baseAt, ProjectionProgram.residue, lcEval,
      Fin.val_add, Fin.val_mul]
  rw [source, ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_add,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul,
    ProductionMessageAcceptance.toConcreteField_mul]

theorem mappedNegOne :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (goldilocksP - 1)) = (-1 : F) := by
  apply Fin.ext
  rfl

theorem mappedNegOne_mul (value : F) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue (goldilocksP - 1)) * value = -value := by
  rw [mappedNegOne]
  calc
    (-1 : F) * value = -(1 * value) := Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

theorem mappedTwo_mul (value : F) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.residue 2) * value = value + value := by
  have mappedTwo : ProductionMessageAcceptance.toConcreteField
      (ProjectionProgram.residue 2) = (2 : F) := by
    apply Fin.ext
    rfl
  rw [mappedTwo]
  have twoEq : (2 : F) = 1 + 1 := by decide
  rw [twoEq]
  calc
    ((1 : F) + 1) * value = value * ((1 : F) + 1) :=
      Lean.Grind.Fin.mul_comm _ _
    _ = value * 1 + value * 1 := Lean.Grind.Fin.left_distrib _ _ _
    _ = value + value := by
      congr 1 <;> exact Lean.Grind.Fin.mul_one value

/-- The sparse terminal equality-factor row computes the standard
multilinear equality factor once its product column is authoritative. -/
theorem affineProduct_eq_equalityFactor (left right : K) :
    K.sub
        (K.sub
          (K.add (K.add (K.mul left right) (K.mul left right)) K.one)
          left)
        right =
      SumCheckTruthPath.equalityFactor ops left right := by
  have negNeg (value : K) : ops.neg (ops.neg value) = value := by
    apply (FiniteSumAlgebra.sub_eq_zero_iff ops laws _ _).mp
    unfold InterpolationOps.sub
    rw [laws.add_comm]
    exact laws.add_neg (ops.neg value)
  rw [← ConcreteCarrier.derived_sub_eq_concrete_sub,
    ← ConcreteCarrier.derived_sub_eq_concrete_sub]
  change ops.sub
      (ops.sub
        (ops.add (ops.add (ops.mul left right) (ops.mul left right)) ops.one)
        left)
      right = SumCheckTruthPath.equalityFactor ops left right
  unfold SumCheckTruthPath.equalityFactor InterpolationOps.sub
  rw [laws.right_distrib, laws.one_mul, laws.left_distrib, laws.mul_one,
    laws.neg_mul, FiniteSumAlgebra.mul_neg ops laws, negNeg]
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  ac_rfl

theorem cubic_sub_eq_range (value : K) :
    K.sub (K.mul (K.mul value value) value) value =
      K.mul (K.mul (K.add value (K.embed 1)) value)
        (K.sub value (K.embed 1)) := by
  have embedOne : K.embed 1 = ops.one := rfl
  rw [embedOne, ← ConcreteCarrier.derived_sub_eq_concrete_sub,
    ← ConcreteCarrier.derived_sub_eq_concrete_sub]
  change ops.sub (ops.mul (ops.mul value value) value) value =
    ops.mul (ops.mul (ops.add value ops.one) value)
      (ops.sub value ops.one)
  unfold InterpolationOps.sub
  let square := ops.mul value value
  let cube := ops.mul square value
  let plus := ops.mul (ops.add value ops.one) value
  have mulNegRight (left right : K) :
      ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
    calc
      ops.mul left (ops.neg right) = ops.mul (ops.neg right) left :=
        laws.mul_comm _ _
      _ = ops.neg (ops.mul right left) := laws.neg_mul _ _
      _ = ops.neg (ops.mul left right) :=
        congrArg ops.neg (laws.mul_comm right left)
  calc
    ops.add cube (ops.neg value) =
        ops.add (ops.add cube square)
          (ops.add (ops.neg square) (ops.neg value)) := by
      rw [laws.add_assoc cube square]
      rw [← laws.add_assoc square (ops.neg square) (ops.neg value)]
      rw [laws.add_neg, laws.zero_add]
    _ = ops.add (ops.add cube square) (ops.neg (ops.add square value)) := by
      rw [laws.neg_add]
    _ = ops.add (ops.mul plus value) (ops.neg plus) := by
      unfold plus square cube
      rw [laws.right_distrib, laws.one_mul, laws.right_distrib]
    _ = ops.add (ops.mul plus value) (ops.mul plus (ops.neg ops.one)) := by
      have mapped := mulNegRight plus ops.one
      rw [laws.mul_one] at mapped
      rw [mapped]
    _ = ops.mul plus (ops.add value (ops.neg ops.one)) := by
      rw [laws.left_distrib]
    _ = ops.mul (ops.mul (ops.add value ops.one) value)
        (ops.add value (ops.neg ops.one)) := rfl

end ProductionTerminalBridge
