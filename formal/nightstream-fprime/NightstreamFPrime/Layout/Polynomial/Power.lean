import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.Polynomial.Power

/-!
Owns the exact physical footprint of the reusable fixed-exponent power
gadget. The first Horner product is multiplication by the constant one and
lowers to two direct affine rows. Each later extension product uses seven
fresh columns and nine rows.

This module proves cost from the symbolic compiler. It does not evaluate a
fixed exponent or own a protocol exponent schedule.
-/

namespace NightstreamFPrime.Layout.Polynomial.Power

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

private theorem isAffine_of_mulCount_zero {expression : Expr}
    (count : R1CS.mulCount expression = 0) : R1CS.IsAffine expression := by
  induction expression with
  | var index => exact R1CS.isAffine_var index
  | const value => exact R1CS.isAffine_const value
  | add left right leftIH rightIH =>
      simp only [R1CS.mulCount, Nat.add_eq_zero_iff] at count
      exact (leftIH count.1).add (rightIH count.2)
  | mul left right leftIH rightIH =>
      simp [R1CS.mulCount] at count

private theorem isAffine_mul_const {expression : Expr}
    (coefficient : F) (affine : R1CS.IsAffine expression) :
    R1CS.IsAffine (expression * .const coefficient) := by
  rcases affine with ⟨lowered, equals⟩
  cases expression <;>
    simp_all [R1CS.IsAffine, R1CS.lowerAffine]

private theorem mulOne_recipesDirect (output : Nat) (point : KExpr)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.RecipesDirect output
      (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes point KExpr.one) := by
  have c0Affine : R1CS.IsAffine point.c0 :=
    isAffine_of_mulCount_zero pointLinear.c0_mulCount
  have c1Affine : R1CS.IsAffine point.c1 :=
    isAffine_of_mulCount_zero pointLinear.c1_mulCount
  have firstAffine : R1CS.IsAffine
      (point.c0 * (1 : Expr) + (7 : Expr) * point.c1 * (0 : Expr)) :=
    (isAffine_mul_const 1 c0Affine).add
      (isAffine_mul_const 0 (R1CS.IsAffine.const_mul 7 c1Affine))
  have secondAffine : R1CS.IsAffine
      (point.c0 * (0 : Expr) + point.c1 * (1 : Expr)) :=
    (isAffine_mul_const 0 c0Affine).add (isAffine_mul_const 1 c1Affine)
  change R1CS.IsDirectRecipe output
      (point.c0 * (1 : Expr) + (7 : Expr) * point.c1 * (0 : Expr)) ∧
    R1CS.IsDirectRecipe (output + 1)
      (point.c0 * (0 : Expr) + point.c1 * (1 : Expr)) ∧ True
  exact ⟨R1CS.IsDirectRecipe.of_affine output firstAffine,
    R1CS.IsDirectRecipe.of_affine (output + 1) secondAffine, trivial⟩

theorem mulOne_totalFreshCount (output : Nat) (point : KExpr)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalFreshCount
      (recipeConstraints output
        (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes point KExpr.one)) =
      0 :=
  R1CS.recipeConstraints_totalFreshCount output _
    (mulOne_recipesDirect output point pointLinear)

theorem mulOne_totalRowCount (output : Nat) (point : KExpr)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalRowCount
      (recipeConstraints output
        (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes point KExpr.one)) =
      2 := by
  rw [R1CS.recipeConstraints_totalRowCount output _
    (mulOne_recipesDirect output point pointLinear)]
  rfl

private theorem coefficientExprs_succ (exponent : Nat) :
    NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs (exponent + 1) =
      KExpr.zero ::
        NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent := by
  simp [NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs,
    List.replicate_succ]

private theorem coefficientExprs_ne_nil (exponent : Nat) :
    NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent ≠ [] := by
  simp [NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs]

private theorem zero_add_product_linear (start : Nat) :
    Horner.KExprLinear
      (KExpr.add KExpr.zero
        (NightstreamFPrime.Gadgets.Polynomial.Horner.productAt start)) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [KExpr.add, KExpr.zero,
      NightstreamFPrime.Gadgets.Polynomial.Horner.productAt,
      R1CS.mulCount, Horner.Nonconstant]

theorem compile_succ_recipes (start : Nat) (point : KExpr)
    (exponent : Nat) :
    (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
      (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
        (exponent + 1))).recipes =
      let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent)
      tail.recipes ++
        NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes point tail.output := by
  rw [coefficientExprs_succ]
  cases coefficientsEquals :
      NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent with
  | nil => exact False.elim (coefficientExprs_ne_nil exponent coefficientsEquals)
  | cons next rest => rfl

theorem compile_succ_output (start : Nat) (point : KExpr)
    (exponent : Nat) :
    (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
      (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
        (exponent + 1))).output =
      let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent)
      KExpr.add KExpr.zero
        (NightstreamFPrime.Gadgets.Polynomial.Horner.productAt
          (start + tail.recipes.length)) := by
  rw [coefficientExprs_succ]
  cases coefficientsEquals :
      NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent with
  | nil => exact False.elim (coefficientExprs_ne_nil exponent coefficientsEquals)
  | cons next rest => rfl

theorem compile_output_linear_succ (start : Nat) (point : KExpr)
    (exponent : Nat) :
    Horner.KExprLinear
      (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
          (exponent + 1))).output := by
  rw [compile_succ_output]
  exact zero_add_product_linear _

theorem totalFreshCount (start : Nat) (point : KExpr) (exponent : Nat)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalFreshCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
            exponent)).recipes) =
      7 * (exponent - 1) := by
  induction exponent generalizing start with
  | zero => rfl
  | succ exponent inductionHypothesis =>
      rw [compile_succ_recipes, recipeConstraints_append,
        R1CS.totalFreshCount_append,
        inductionHypothesis (start := start)]
      cases exponent with
      | zero =>
          change 0 + R1CS.totalFreshCount
            (recipeConstraints start
              (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes
                point KExpr.one)) = _
          rw [mulOne_totalFreshCount start point pointLinear]
      | succ exponent =>
          rw [Horner.mulRecipes_totalFreshCount _ point _ pointLinear
            (compile_output_linear_succ start point exponent)]
          omega

theorem totalRowCount (start : Nat) (point : KExpr) (exponent : Nat)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalRowCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
            exponent)).recipes) =
      if exponent = 0 then 0 else 9 * exponent - 7 := by
  induction exponent generalizing start with
  | zero => rfl
  | succ exponent inductionHypothesis =>
      rw [compile_succ_recipes, recipeConstraints_append,
        R1CS.totalRowCount_append,
        inductionHypothesis (start := start)]
      cases exponent with
      | zero =>
          change 0 + R1CS.totalRowCount
            (recipeConstraints start
              (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes
                point KExpr.one)) = _
          rw [mulOne_totalRowCount start point pointLinear]
          simp
      | succ exponent =>
          rw [Horner.mulRecipes_totalRowCount _ point _ pointLinear
            (compile_output_linear_succ start point exponent)]
          simp
          omega

theorem ownedCircuit_totalFreshCount (exponent : Nat)
    (interface : NightstreamFPrime.Gadgets.Polynomial.Power.Interface)
    (offset : Nat) (pointLinear : Horner.KExprLinear (interface.point offset)) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Polynomial.Power.circuit exponent interface
        ).main offset)) =
      7 * (exponent - 1) := by
  unfold NightstreamFPrime.Gadgets.Polynomial.Power.circuit
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact totalFreshCount offset (interface.point offset) exponent pointLinear

theorem ownedCircuit_totalRowCount (exponent : Nat)
    (interface : NightstreamFPrime.Gadgets.Polynomial.Power.Interface)
    (offset : Nat) (pointLinear : Horner.KExprLinear (interface.point offset)) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Polynomial.Power.circuit exponent interface
        ).main offset)) =
      if exponent = 0 then 0 else 9 * exponent - 7 := by
  unfold NightstreamFPrime.Gadgets.Polynomial.Power.circuit
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact totalRowCount offset (interface.point offset) exponent pointLinear

end NightstreamFPrime.Layout.Polynomial.Power
