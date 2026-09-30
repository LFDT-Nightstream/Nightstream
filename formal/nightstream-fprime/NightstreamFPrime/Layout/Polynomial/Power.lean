import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.Polynomial.Power

/-!
Owns the exact physical footprint of the reusable fixed-exponent power
gadget. Each stored extension product is three rank-one rows and needs no
lowering cell.

This module proves cost from the symbolic compiler. It does not evaluate a
fixed exponent or own a protocol exponent schedule.
-/

namespace NightstreamFPrime.Layout.Polynomial.Power

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

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

private theorem coefficientExprs_kAffine (exponent : Nat) :
    ∀ coefficient ∈
      NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs exponent,
      Horner.KAffine coefficient := by
  intro coefficient member
  simp only [NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs,
    List.mem_append, List.mem_replicate, List.mem_singleton] at member
  rcases member with ⟨_, rfl⟩ | rfl <;>
    exact ⟨R1CS.isAffine_const _, R1CS.isAffine_const _⟩

private theorem recipesDirect (start : Nat) (point : KExpr) (exponent : Nat)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.RecipesDirect start
      (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
          exponent)).recipes :=
  Horner.compile_recipesDirect start point _ pointLinear.kAffine
    (coefficientExprs_kAffine exponent)

theorem totalFreshCount (start : Nat) (point : KExpr) (exponent : Nat)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalFreshCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
            exponent)).recipes) = 0 :=
  R1CS.recipeConstraints_totalFreshCount start _
    (recipesDirect start point exponent pointLinear)

theorem totalRowCount (start : Nat) (point : KExpr) (exponent : Nat)
    (pointLinear : Horner.KExprLinear point) :
    R1CS.totalRowCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          (NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs
            exponent)).recipes) = 3 * exponent := by
  rw [R1CS.recipeConstraints_totalRowCount start _
      (recipesDirect start point exponent pointLinear),
    NightstreamFPrime.Gadgets.Polynomial.Horner.compile_recipes_length,
    NightstreamFPrime.Gadgets.Polynomial.Power.coefficientExprs_length]
  omega

theorem ownedCircuit_totalFreshCount (exponent : Nat)
    (interface : NightstreamFPrime.Gadgets.Polynomial.Power.Interface)
    (offset : Nat) (pointLinear : Horner.KExprLinear (interface.point offset)) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Polynomial.Power.circuit exponent interface
        ).main offset)) = 0 := by
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
        ).main offset)) = 3 * exponent := by
  unfold NightstreamFPrime.Gadgets.Polynomial.Power.circuit
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact totalRowCount offset (interface.point offset) exponent pointLinear

end NightstreamFPrime.Layout.Polynomial.Power
