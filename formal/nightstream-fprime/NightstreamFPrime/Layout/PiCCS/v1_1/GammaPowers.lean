import NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers
import NightstreamFPrime.Layout.Polynomial.Horner

/-! Exact costs when the shared gamma input is the two transcript wires. -/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers
open NightstreamFPrime.Layout.Polynomial.Horner

private def pairs (gamma : KExpr) (start : Nat) : List (KExpr × KExpr) :=
  List.ofFn fun step : Fin 16 =>
    (wire gamma start (arguments step).1, wire gamma start (arguments step).2)

private theorem recipes_eq_pairs (gamma : KExpr) (start : Nat) :
    recipes gamma start = (pairs gamma start).flatMap fun pair =>
      Gadgets.Polynomial.Horner.mulRecipes pair.1 pair.2 := by rfl

private theorem pair_costs (start : Nat) (values : List (KExpr × KExpr))
    (linear : ∀ pair ∈ values, KExprLinear pair.1 ∧ KExprLinear pair.2) :
    R1CS.totalFreshCount (recipeConstraints start
      (values.flatMap fun pair => Gadgets.Polynomial.Horner.mulRecipes pair.1 pair.2)) =
        7 * values.length ∧
    R1CS.totalRowCount (recipeConstraints start
      (values.flatMap fun pair => Gadgets.Polynomial.Horner.mulRecipes pair.1 pair.2)) =
        9 * values.length := by
  induction values generalizing start with
  | nil => exact ⟨rfl, rfl⟩
  | cons pair values ih =>
      have head := linear pair (by simp)
      have tail := ih (start + 2) (fun value member => linear value (by simp [member]))
      simp only [List.flatMap_cons, recipeConstraints_append,
        R1CS.totalFreshCount_append, R1CS.totalRowCount_append,
        Gadgets.Polynomial.Horner.mulRecipes_length]
      rw [mulRecipes_totalFreshCount start pair.1 pair.2 head.1 head.2,
        mulRecipes_totalRowCount start pair.1 pair.2 head.1 head.2,
        tail.1, tail.2]
      simp [Nat.mul_add, Nat.add_comm]

private theorem wire_linear (gamma : KExpr) (start index : Nat)
    (linear : KExprLinear gamma) : KExprLinear (wire gamma start index) := by
  cases index with
  | zero => exact linear
  | succ index => exact productAt_linear _

theorem input_costs (gamma : KExpr) (start : Nat) (gammaLinear : KExprLinear gamma) :
    R1CS.totalFreshCount (recipeConstraints start
      (recipes gamma start)) = 112 ∧
    R1CS.totalRowCount (recipeConstraints start
      (recipes gamma start)) = 144 := by
  rw [recipes_eq_pairs]
  have costs := pair_costs start (pairs gamma start) (by
    intro pair member
    rw [pairs, List.mem_ofFn'] at member
    obtain ⟨step, rfl⟩ := member
    exact ⟨wire_linear _ _ _ gammaLinear, wire_linear _ _ _ gammaLinear⟩)
  simpa only [pairs, List.length_ofFn] using costs

theorem transcript_wire_costs (first second start : Nat) :
    R1CS.totalFreshCount (recipeConstraints start
      (recipes ⟨Expr.var first, Expr.var second⟩ start)) = 112 ∧
    R1CS.totalRowCount (recipeConstraints start
      (recipes ⟨Expr.var first, Expr.var second⟩ start)) = 144 := by
  apply input_costs
  refine ⟨rfl, rfl, ?_, ?_⟩ <;> simp [Nonconstant]

theorem local_coordinate_count : (32 + 112) * 41 = 5904 := by decide

theorem local_savings :
    124402 - 144 = 124258 ∧ 5100482 - 5904 = 5094578 := by decide

end NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers
