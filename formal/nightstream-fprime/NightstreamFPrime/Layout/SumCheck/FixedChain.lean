import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.SumCheck.FixedChain

/-!
Owns physical R1CS cost proofs for the reusable SumCheck chain with stored
round evaluations. It counts one generic round and composes that result by
list induction. It does not own transcript challenges, protocol round count,
or terminal checks.
-/

namespace NightstreamFPrime.Layout.SumCheck.FixedChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck.FixedChain
open NightstreamFPrime.Layout.Polynomial.Horner

/-- Every coefficient and the challenge of one round are plain wires or sums
of wires. This is the syntactic boundary for stable round costs. -/
structure RoundLinear {degree : Nat} (round : Round degree) : Prop where
  coefficient : ∀ index, KExprLinear (round.coefficient index)
  challenge : KExprLinear round.challenge

private theorem coefficients_linear {degree : Nat} (round : Round degree)
    (linear : RoundLinear round) :
    ∀ coefficient ∈ round.coefficients, KExprLinear coefficient := by
  intro coefficient member
  rw [Round.coefficients, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact linear.coefficient index

private theorem coefficients_length {degree : Nat} (round : Round degree) :
    round.coefficients.length = degree + 1 := by
  simp [Round.coefficients]

private theorem coefficients_nonempty {degree : Nat} (round : Round degree) :
    round.coefficients ≠ [] := by
  intro empty
  have length := coefficients_length round
  rw [empty] at length
  simp at length

/-- The stored `p(r)` is a sum of wires for the next round and the terminal
owner. -/
theorem roundProgram_output_linear {degree : Nat} (start : Nat)
    (round : Round degree) (linear : RoundLinear round) :
    KExprLinear (Owned.roundProgram start round).output :=
  compile_output_linear start round.challenge round.coefficients
    (coefficients_nonempty round) (coefficients_linear round linear)

theorem roundProgram_totalFreshCount {degree : Nat} (start : Nat)
    (round : Round degree) (linear : RoundLinear round) :
    R1CS.totalFreshCount
      (recipeConstraints start (Owned.roundProgram start round).recipes) =
      0 := by
  rw [Owned.roundProgram, compile_totalFreshCount start round.challenge
    round.coefficients linear.challenge (coefficients_linear round linear)]

theorem recipesFrom_totalFreshCount {degree : Nat} (start : Nat)
    (rounds : List (Round degree))
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalFreshCount
      (recipeConstraints start (Owned.recipesFrom start rounds)) = 0 := by
  induction rounds generalizing start with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      rw [Owned.recipesFrom, recipeConstraints_append,
        R1CS.totalFreshCount_append,
        roundProgram_totalFreshCount start round (linear round (by simp)),
        Owned.roundProgram_recipes_length,
        inductionHypothesis (start + 3 * degree)
          (fun later member => linear later (by simp [member]))]

private theorem sub_affine {left right : Expr}
    (leftAffine : R1CS.IsAffine left) (rightAffine : R1CS.IsAffine right) :
    R1CS.IsAffine (left - right) :=
  R1CS.IsAffine.add leftAffine (R1CS.IsAffine.const_mul (-1) rightAffine)

private theorem coefficientSum_affine (coefficients : List KExpr)
    (linear : ∀ coefficient ∈ coefficients, KExprLinear coefficient) :
    R1CS.IsAffine (Owned.coefficientSum coefficients).c0 ∧
      R1CS.IsAffine (Owned.coefficientSum coefficients).c1 := by
  induction coefficients with
  | nil => exact ⟨R1CS.isAffine_const 0, R1CS.isAffine_const 0⟩
  | cons coefficient rest inductionHypothesis =>
      have head := (linear coefficient (by simp)).isAffine
      have tail := inductionHypothesis fun current member =>
        linear current (by simp [member])
      exact ⟨R1CS.IsAffine.add head.1 tail.1,
        R1CS.IsAffine.add head.2 tail.2⟩

private theorem boundarySum_affine (coefficients : List KExpr)
    (linear : ∀ coefficient ∈ coefficients, KExprLinear coefficient) :
    R1CS.IsAffine (Owned.boundarySum coefficients).c0 ∧
      R1CS.IsAffine (Owned.boundarySum coefficients).c1 := by
  cases coefficients with
  | nil =>
      exact ⟨R1CS.IsAffine.add (R1CS.isAffine_const 0) (R1CS.isAffine_const 0),
        R1CS.IsAffine.add (R1CS.isAffine_const 0) (R1CS.isAffine_const 0)⟩
  | cons coefficient rest =>
      have head := (linear coefficient (by simp)).isAffine
      have sum := coefficientSum_affine (coefficient :: rest) linear
      exact ⟨R1CS.IsAffine.add head.1 sum.1, R1CS.IsAffine.add head.2 sum.2⟩

/-- A round equation reads only wires, so it lowers to one direct row. -/
theorem roundEqualities_affine {degree : Nat} (current : KExpr)
    (round : Round degree) (currentLinear : KExprLinear current)
    (linear : RoundLinear round) :
    ∀ expression ∈ KExpr.equalities current (Owned.roundBoundary round),
      R1CS.IsAffine expression := by
  have current := currentLinear.isAffine
  have boundary := boundarySum_affine round.coefficients
    (coefficients_linear round linear)
  intro expression member
  simp only [KExpr.equalities, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact sub_affine current.1 boundary.1
  · exact sub_affine current.2 boundary.2

theorem equalitiesFrom_affine {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) (currentLinear : KExprLinear current)
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    ∀ expression ∈ Owned.equalitiesFrom start current rounds,
      R1CS.IsAffine expression := by
  induction rounds generalizing start current with
  | nil =>
      intro expression member
      simp [Owned.equalitiesFrom] at member
  | cons round rounds inductionHypothesis =>
      intro expression member
      rw [Owned.equalitiesFrom] at member
      rcases List.mem_append.mp member with headMember | tailMember
      · exact roundEqualities_affine current round currentLinear
          (linear round (by simp)) expression headMember
      · exact inductionHypothesis (start + 3 * degree)
          (Owned.roundProgram start round).output
          (roundProgram_output_linear start round (linear round (by simp)))
          (fun later laterMember => linear later (by simp [laterMember]))
          expression tailMember

theorem outputFrom_linear {degree : Nat} (start : Nat) (current : KExpr)
    (rounds : List (Round degree)) (currentLinear : KExprLinear current)
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    KExprLinear (Owned.outputFrom start current rounds) := by
  induction rounds generalizing start current with
  | nil => exact currentLinear
  | cons round rounds inductionHypothesis =>
      exact inductionHypothesis (start + 3 * degree)
        (Owned.roundProgram start round).output
        (roundProgram_output_linear start round (linear round (by simp)))
        (fun later member => linear later (by simp [member]))

private theorem interfaceRounds_linear {degree roundCount : Nat}
    (interface : Owned.Interface degree roundCount)
    (linear : ∀ round, RoundLinear (interface.round round)) :
    ∀ round ∈ interface.rounds, RoundLinear round := by
  intro round member
  rw [Owned.Interface.rounds, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact linear index

/-- The stored products and the round equations need no lowering cell. -/
theorem ownedCircuit_totalFreshCount {degree roundCount : Nat}
    (interface : Owned.Interface degree roundCount) (offset : Nat)
    (initialLinear : KExprLinear interface.initial)
    (linear : ∀ round, RoundLinear (interface.round round)) :
    R1CS.totalFreshCount (flatConstraints
      (Circuit.ops (Owned.circuit interface).main offset)) = 0 := by
  have equalityFresh : R1CS.totalFreshCount
      (Owned.assertions interface offset) = 0 :=
    R1CS.totalFreshCount_eq_zero_of_noFresh _ fun expression member =>
      R1CS.constraintFreshCount_eq_zero_of_affine expression
        (equalitiesFrom_affine offset interface.initial interface.rounds
          initialLinear (interfaceRounds_linear interface linear)
          expression member)
  rw [Owned.flatConstraints_eq, R1CS.totalFreshCount_append, equalityFresh,
    Owned.recipes, recipesFrom_totalFreshCount offset interface.rounds
      (interfaceRounds_linear interface linear)]

/-- Each stored product costs three rows, and each round equation two. -/
theorem ownedCircuit_totalRowCount {degree roundCount : Nat}
    (interface : Owned.Interface degree roundCount) (offset : Nat)
    (initialLinear : KExprLinear interface.initial)
    (linear : ∀ round, RoundLinear (interface.round round)) :
    R1CS.totalRowCount (flatConstraints
      (Circuit.ops (Owned.circuit interface).main offset)) =
      3 * degree * roundCount + 2 * roundCount := by
  rw [R1CS.totalRowCount_eq_fresh_add_length,
    ownedCircuit_totalFreshCount interface offset initialLinear linear,
    Owned.flatConstraints_length, Owned.privateCount]
  omega

/-- The owned final claim is a sum of wires for the terminal owner. -/
theorem ownedOutput_linear {degree roundCount : Nat}
    (interface : Owned.Interface degree roundCount) (offset : Nat)
    (initialLinear : KExprLinear interface.initial)
    (linear : ∀ round, RoundLinear (interface.round round)) :
    KExprLinear (Owned.output interface offset) :=
  outputFrom_linear offset interface.initial interface.rounds initialLinear
    (interfaceRounds_linear interface linear)

end NightstreamFPrime.Layout.SumCheck.FixedChain
