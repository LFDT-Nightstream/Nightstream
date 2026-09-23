import NightstreamFPrime.Gadgets.SumCheck.CompactChain
import NightstreamFPrime.Layout.Polynomial.Horner

/-! Exact R1CS costs for the materialized SumCheck chain. -/

namespace NightstreamFPrime.Layout.SumCheck.CompactChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Layout.Polynomial.Horner


structure RoundLinear {degree : Nat} (round : FixedChain.Round degree) : Prop where
  coefficient : ∀ index, KExprLinear (round.coefficient index)
  challenge : KExprLinear round.challenge

private theorem coefficients_linear {degree : Nat} (round : FixedChain.Round degree)
    (linear : RoundLinear round) :
    ∀ coefficient ∈ round.coefficients, KExprLinear coefficient := by
  intro coefficient member
  rw [FixedChain.Round.coefficients, List.mem_ofFn'] at member
  obtain ⟨index, rfl⟩ := member
  exact linear.coefficient index

private theorem coefficientSum_noMul (coefficients : List KExpr)
    (linear : ∀ coefficient ∈ coefficients, KExprLinear coefficient) :
    R1CS.mulCount (Gadgets.SumCheck.CompactChain.coefficientSum coefficients).c0 = 0 ∧
      R1CS.mulCount (Gadgets.SumCheck.CompactChain.coefficientSum coefficients).c1 = 0 := by
  induction coefficients with
  | nil => exact ⟨rfl, rfl⟩
  | cons coefficient rest ih =>
      have head := linear coefficient (by simp)
      have tail := ih (fun value member => linear value (by simp [member]))
      simp [Gadgets.SumCheck.CompactChain.coefficientSum, KExpr.add, R1CS.mulCount,
        head.c0_mulCount, head.c1_mulCount, tail.1, tail.2]

private theorem booleanSum_affine (coefficients : List KExpr)
    (linear : ∀ coefficient ∈ coefficients, KExprLinear coefficient) :
    R1CS.IsAffine (Gadgets.SumCheck.CompactChain.booleanSum coefficients).c0 ∧
      R1CS.IsAffine (Gadgets.SumCheck.CompactChain.booleanSum coefficients).c1 := by
  have tail := coefficientSum_noMul coefficients linear
  have head : R1CS.mulCount (coefficients.headD KExpr.zero).c0 = 0 ∧
      R1CS.mulCount (coefficients.headD KExpr.zero).c1 = 0 := by
    cases coefficients with
    | nil => exact ⟨rfl, rfl⟩
    | cons coefficient rest =>
        exact ⟨(linear coefficient (by simp)).c0_mulCount,
          (linear coefficient (by simp)).c1_mulCount⟩
  constructor <;> apply isAffine_of_mulCount_zero <;>
    simp only [Gadgets.SumCheck.CompactChain.booleanSum, KExpr.add, R1CS.mulCount,
      head.1, head.2, tail.1, tail.2, Nat.zero_add]

private theorem equalities_affine (current : KExpr) (coefficients : List KExpr)
    (currentLinear : KExprLinear current)
    (linear : ∀ coefficient ∈ coefficients, KExprLinear coefficient) :
    ∀ expression ∈ KExpr.equalities current (Gadgets.SumCheck.CompactChain.booleanSum coefficients),
      R1CS.IsAffine expression := by
  have right := booleanSum_affine coefficients linear
  intro expression member
  simp only [KExpr.equalities, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact currentLinear.isAffine.1.add (right.1.const_mul (-1))
  · exact currentLinear.isAffine.2.add (right.2.const_mul (-1))

theorem compile_output_linear {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (FixedChain.Round degree))
    (currentLinear : KExprLinear current)
    (roundsLinear : ∀ round ∈ rounds, RoundLinear round) :
    KExprLinear (Gadgets.SumCheck.CompactChain.compile start current rounds).output := by
  induction rounds generalizing start current with
  | nil => exact currentLinear
  | cons round rounds ih =>
      have linear := roundsLinear round (by simp)
      have evaluationLinear := Polynomial.Horner.compile_output_linear start
        round.challenge round.coefficients (by simp [FixedChain.Round.coefficients])
        (coefficients_linear round linear)
      exact ih _ _ evaluationLinear (fun later member =>
        roundsLinear later (by simp [member]))

theorem compile_costs {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (FixedChain.Round degree))
    (currentLinear : KExprLinear current)
    (roundsLinear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalFreshCount
        (recipeConstraints start (Gadgets.SumCheck.CompactChain.compile start current rounds).recipes) =
        7 * degree * rounds.length ∧
      R1CS.totalRowCount
        (recipeConstraints start (Gadgets.SumCheck.CompactChain.compile start current rounds).recipes) =
        9 * degree * rounds.length ∧
      R1CS.totalFreshCount (Gadgets.SumCheck.CompactChain.compile start current rounds).checks = 0 ∧
      R1CS.totalRowCount (Gadgets.SumCheck.CompactChain.compile start current rounds).checks =
        2 * rounds.length := by
  induction rounds generalizing start current with
  | nil => simp [Gadgets.SumCheck.CompactChain.compile, recipeConstraints, R1CS.totalFreshCount, R1CS.totalRowCount]
  | cons round rounds ih =>
      have linear := roundsLinear round (by simp)
      have coefficientsLinear := coefficients_linear round linear
      have evaluationLinear := Polynomial.Horner.compile_output_linear start
        round.challenge round.coefficients (by simp [FixedChain.Round.coefficients])
        coefficientsLinear
      have tail := ih
        (start + (Gadgets.Polynomial.Horner.compile start round.challenge round.coefficients).recipes.length)
        (Gadgets.Polynomial.Horner.compile start round.challenge round.coefficients).output
        evaluationLinear (fun later member =>
        roundsLinear later (by simp [member]))
      have fresh := Polynomial.Horner.compile_totalFreshCount start round.challenge
        round.coefficients linear.challenge coefficientsLinear
      have rows := Polynomial.Horner.compile_totalRowCount start round.challenge
        round.coefficients linear.challenge coefficientsLinear
      have headAffine := equalities_affine current round.coefficients
        currentLinear coefficientsLinear
      have headFresh := R1CS.totalFreshCount_eq_zero_of_noFresh _ (fun expression member =>
        R1CS.constraintFreshCount_eq_zero_of_affine expression (headAffine expression member))
      have headRows := R1CS.totalRowCount_eq_length_of_rowsOne _ (fun expression member =>
        R1CS.constraintRowCount_eq_one_of_affine expression (headAffine expression member))
      simp only [Gadgets.SumCheck.CompactChain.compile_cons, recipeConstraints_append, R1CS.totalFreshCount_append,
        R1CS.totalRowCount_append, List.length_cons]
      rw [fresh, rows, tail.1, tail.2.1, headFresh, tail.2.2.1,
        headRows, tail.2.2.2]
      simp [FixedChain.Round.coefficients, KExpr.equalities, Nat.mul_add, Nat.add_comm]

theorem production_costs (interface : Gadgets.SumCheck.CompactChain.Interface 9 28) (offset : Nat)
    (initialLinear : KExprLinear interface.initial)
    (roundsLinear : ∀ index, RoundLinear (interface.round index)) :
    localLength (Circuit.ops (Gadgets.SumCheck.CompactChain.circuit interface).main offset) = 504 ∧
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops (Gadgets.SumCheck.CompactChain.circuit interface).main offset)) = 1764 ∧
      R1CS.totalRowCount
        (flatConstraints (Circuit.ops (Gadgets.SumCheck.CompactChain.circuit interface).main offset)) = 2324 := by
  have costs := compile_costs offset interface.initial interface.rounds initialLinear (by
    intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    obtain ⟨index, rfl⟩ := member
    exact roundsLinear index)
  change localLength (Circuit.ops (Gadgets.SumCheck.CompactChain.main interface) offset) = _ ∧ _
  rw [Gadgets.SumCheck.CompactChain.localLength_eq]
  change (Gadgets.SumCheck.CompactChain.compile offset interface.initial interface.rounds).recipes.length = _ ∧ _
  rw [Gadgets.SumCheck.CompactChain.compile_recipes_length]
  change _ ∧ R1CS.totalFreshCount (flatConstraints (Gadgets.SumCheck.CompactChain.opsAt interface offset)) = _ ∧
    R1CS.totalRowCount (flatConstraints (Gadgets.SumCheck.CompactChain.opsAt interface offset)) = _
  rw [Gadgets.SumCheck.CompactChain.flatConstraints_opsAt, R1CS.totalFreshCount_append,
    R1CS.totalRowCount_append]
  change _ ∧ R1CS.totalFreshCount (recipeConstraints offset
    (Gadgets.SumCheck.CompactChain.compile offset interface.initial interface.rounds).recipes) +
    R1CS.totalFreshCount (Gadgets.SumCheck.CompactChain.compile offset interface.initial interface.rounds).checks = _ ∧ _
  simp only [Gadgets.SumCheck.CompactChain.program]
  rw [costs.1, costs.2.1, costs.2.2.1, costs.2.2.2]
  simp [FixedChain.Owned.Interface.rounds]

end NightstreamFPrime.Layout.SumCheck.CompactChain
