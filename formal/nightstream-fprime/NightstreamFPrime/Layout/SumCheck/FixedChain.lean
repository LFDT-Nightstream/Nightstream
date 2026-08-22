import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.SumCheck.FixedChain

/-!
Owns physical R1CS cost proofs for the reusable degree-4 SumCheck chain.
It counts one generic round and composes that result by list induction. It
does not own transcript challenges, protocol round count, or terminal checks.
-/

namespace NightstreamFPrime.Layout.SumCheck.FixedChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Layout.Polynomial.Horner

def KExprMulCount (value : KExpr) : Nat :=
  R1CS.mulCount value.c0 + R1CS.mulCount value.c1

def evaluationCounts : Nat → Nat × Nat
  | 0 => (0, 0)
  | count + 1 =>
      let previous := evaluationCounts count
      (previous.1 + previous.2 + 3,
        previous.1 + previous.2 + 2)

theorem evaluateCoefficients_mulCounts
    (point : KExpr) (coefficients : List KExpr)
    (pointNoMul : R1CS.mulCount point.c0 = 0 ∧
      R1CS.mulCount point.c1 = 0)
    (coefficientsNoMul : ∀ coefficient ∈ coefficients,
      R1CS.mulCount coefficient.c0 = 0 ∧
        R1CS.mulCount coefficient.c1 = 0) :
    R1CS.mulCount
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          point coefficients).c0 = (evaluationCounts coefficients.length).1 ∧
      R1CS.mulCount
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          point coefficients).c1 = (evaluationCounts coefficients.length).2 := by
  induction coefficients with
  | nil => exact ⟨rfl, rfl⟩
  | cons coefficient coefficients inductionHypothesis =>
      have headNoMul := coefficientsNoMul coefficient (by simp)
      have tailNoMul : ∀ current ∈ coefficients,
          R1CS.mulCount current.c0 = 0 ∧
            R1CS.mulCount current.c1 = 0 := by
        intro current member
        exact coefficientsNoMul current (by simp [member])
      have tailCounts := inductionHypothesis tailNoMul
      simp only [NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients,
        KExpr.add, KExpr.mul, R1CS.mulCount, evaluationCounts,
        headNoMul.1, headNoMul.2, pointNoMul.1,
        pointNoMul.2, tailCounts.1, tailCounts.2]
      omega

structure RoundLinear
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4) : Prop where
  coefficient : ∀ index, KExprLinear (round.coefficient index)
  challenge : KExprLinear round.challenge

private theorem roundCoefficients_noMul
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (linear : RoundLinear round) :
    ∀ coefficient ∈ round.coefficients,
      R1CS.mulCount coefficient.c0 = 0 ∧
        R1CS.mulCount coefficient.c1 = 0 := by
  intro coefficient member
  rw [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.coefficients,
    List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact ⟨(linear.coefficient index).c0_mulCount,
    (linear.coefficient index).c1_mulCount⟩

theorem evaluateRound_mulCounts
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (point : KExpr)
    (pointNoMul : R1CS.mulCount point.c0 = 0 ∧
      R1CS.mulCount point.c1 = 0)
    (linear : RoundLinear round) :
    R1CS.mulCount
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round point).c0 = 78 ∧
      R1CS.mulCount
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round point).c1 = 77 := by
  unfold NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
  have counts := evaluateCoefficients_mulCounts point round.coefficients
    pointNoMul (roundCoefficients_noMul round linear)
  simpa [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.coefficients,
    evaluationCounts] using counts

theorem evaluateRound_totalMulCount
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (linear : RoundLinear round) :
    KExprMulCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round round.challenge) = 155 := by
  have counts := evaluateRound_mulCounts round round.challenge
    ⟨linear.challenge.c0_mulCount, linear.challenge.c1_mulCount⟩ linear
  simp [KExprMulCount, counts.1, counts.2]

private theorem evaluateCoefficients_one_lowerAffine_none_of_three
    (first second third : KExpr) (rest : List KExpr) :
    R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          KExpr.one (first :: second :: third :: rest)).c0 = none ∧
      R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          KExpr.one (first :: second :: third :: rest)).c1 = none := by
  simp [
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients,
    KExpr.add, KExpr.mul, KExpr.one, R1CS.lowerAffine]

private theorem evaluateCoefficients_zero_lowerAffine_none_of_three
    (first second third : KExpr) (rest : List KExpr) :
    R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          KExpr.zero (first :: second :: third :: rest)).c0 = none ∧
      R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients
          KExpr.zero (first :: second :: third :: rest)).c1 = none := by
  simp [
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateCoefficients,
    KExpr.add, KExpr.mul, KExpr.zero, R1CS.lowerAffine]

private theorem evaluateRound_one_lowerAffine_none
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4) :
    R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round KExpr.one).c0 = none ∧
      R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round KExpr.one).c1 = none := by
  generalize coefficientsEq : round.coefficients = coefficients
  have coefficientsLength : coefficients.length = 5 := by
    rw [← coefficientsEq]
    simp [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.coefficients]
  cases coefficients with
  | nil => simp at coefficientsLength
  | cons first coefficients =>
      cases coefficients with
      | nil => simp at coefficientsLength
      | cons second coefficients =>
          cases coefficients with
          | nil => simp at coefficientsLength
          | cons third rest =>
              unfold NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
              rw [coefficientsEq]
              exact evaluateCoefficients_one_lowerAffine_none_of_three
                first second third rest

private theorem evaluateRound_zero_lowerAffine_none
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4) :
    R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round KExpr.zero).c0 = none ∧
      R1CS.lowerAffine
        (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
          round KExpr.zero).c1 = none := by
  generalize coefficientsEq : round.coefficients = coefficients
  have coefficientsLength : coefficients.length = 5 := by
    rw [← coefficientsEq]
    simp [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.coefficients]
  cases coefficients with
  | nil => simp at coefficientsLength
  | cons first coefficients =>
      cases coefficients with
      | nil => simp at coefficientsLength
      | cons second coefficients =>
          cases coefficients with
          | nil => simp at coefficientsLength
          | cons third rest =>
              unfold NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
              rw [coefficientsEq]
              exact evaluateCoefficients_zero_lowerAffine_none_of_three
                first second third rest

private theorem affineConstraint_sub_add_eq_none
    (left first second : Expr)
    (firstNone : R1CS.lowerAffine first = none) :
    R1CS.affineConstraint (left - (first + second)) = none := by
  have rightNone :
      R1CS.lowerAffine (.add first second) = none := by
    unfold R1CS.lowerAffine
    rw [firstNone]
  have negativeNone :
      R1CS.lowerAffine
        (.mul (.const (-1)) (.add first second)) = none := by
    unfold R1CS.lowerAffine
    rw [rightNone]
  have wholeNone :
      R1CS.lowerAffine
        (.add left (.mul (.const (-1)) (.add first second))) = none := by
    cases leftResult : R1CS.lowerAffine left with
    | none =>
        unfold R1CS.lowerAffine
        rw [leftResult]
    | some lowered =>
        unfold R1CS.lowerAffine
        rw [leftResult, negativeNone]
  unfold R1CS.affineConstraint
  rw [show left - (first + second) =
    .add left (.mul (.const (-1)) (.add first second)) by rfl]
  rw [wholeNone]

private theorem directConstraint_sub_add_right_eq_none
    (left first second : Expr)
    (firstNone : R1CS.lowerAffine first = none) :
    R1CS.directConstraint (left - (first + second)) = none := by
  cases left with
  | var index =>
      exact
        NightstreamFPrime.Layout.Polynomial.Horner.directConstraint_sub_add_eq_none
          index first second firstNone
  | const value =>
      exact affineConstraint_sub_add_eq_none (.const value) first second
        firstNone
  | add left right =>
      exact affineConstraint_sub_add_eq_none (.add left right) first second
        firstNone
  | mul left right =>
      exact affineConstraint_sub_add_eq_none (.mul left right) first second
        firstNone

private theorem roundRight_mulCounts
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (linear : RoundLinear round) :
    let right := KExpr.add
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.zero)
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.one)
    R1CS.mulCount right.c0 = 156 ∧
      R1CS.mulCount right.c1 = 154 := by
  dsimp only
  have zeroCounts := evaluateRound_mulCounts round KExpr.zero
    ⟨rfl, rfl⟩ linear
  have oneCounts := evaluateRound_mulCounts round KExpr.one
    ⟨rfl, rfl⟩ linear
  simp [KExpr.add, R1CS.mulCount, zeroCounts.1, zeroCounts.2,
    oneCounts.1, oneCounts.2]

private theorem equalityFreshCount
    (left first second : Expr)
    (firstNone : R1CS.lowerAffine first = none) :
    R1CS.constraintFreshCount (left - (first + second)) =
      R1CS.mulCount left + R1CS.mulCount first +
        R1CS.mulCount second + 1 := by
  unfold R1CS.constraintFreshCount
  rw [directConstraint_sub_add_right_eq_none left first second firstNone]
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) (.add first second))) = _
  simp only [R1CS.mulCount]
  omega

private theorem equalityRowCount
    (left first second : Expr)
    (firstNone : R1CS.lowerAffine first = none) :
    R1CS.constraintRowCount (left - (first + second)) =
      R1CS.mulCount left + R1CS.mulCount first +
        R1CS.mulCount second + 2 := by
  unfold R1CS.constraintRowCount
  rw [directConstraint_sub_add_right_eq_none left first second firstNone]
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) (.add first second))) + 1 = _
  simp only [R1CS.mulCount]
  omega

theorem roundEqualities_totalFreshCount
    (current : KExpr)
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (linear : RoundLinear round) :
    let right := KExpr.add
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.zero)
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.one)
    R1CS.totalFreshCount (KExpr.equalities current right) =
      KExprMulCount current + 312 := by
  dsimp only
  have firstNone := evaluateRound_zero_lowerAffine_none round
  have rightCounts := roundRight_mulCounts round linear
  dsimp only at rightCounts
  have rightC0 :
      R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.zero).c0 +
        R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.one).c0 = 156 := by
    simpa [KExpr.add, R1CS.mulCount] using rightCounts.1
  have rightC1 :
      R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.zero).c1 +
        R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.one).c1 = 154 := by
    simpa [KExpr.add, R1CS.mulCount] using rightCounts.2
  simp only [KExpr.equalities, R1CS.totalFreshCount, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, KExpr.add]
  rw [equalityFreshCount _ _ _ firstNone.1,
    equalityFreshCount _ _ _ firstNone.2]
  unfold KExprMulCount
  omega

theorem roundEqualities_totalRowCount
    (current : KExpr)
    (round : NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4)
    (linear : RoundLinear round) :
    let right := KExpr.add
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.zero)
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
        round KExpr.one)
    R1CS.totalRowCount (KExpr.equalities current right) =
      KExprMulCount current + 314 := by
  dsimp only
  have firstNone := evaluateRound_zero_lowerAffine_none round
  have rightCounts := roundRight_mulCounts round linear
  dsimp only at rightCounts
  have rightC0 :
      R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.zero).c0 +
        R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.one).c0 = 156 := by
    simpa [KExpr.add, R1CS.mulCount] using rightCounts.1
  have rightC1 :
      R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.zero).c1 +
        R1CS.mulCount
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round KExpr.one).c1 = 154 := by
    simpa [KExpr.add, R1CS.mulCount] using rightCounts.2
  simp only [KExpr.equalities, R1CS.totalRowCount, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, KExpr.add]
  rw [equalityRowCount _ _ _ firstNone.1,
    equalityRowCount _ _ _ firstNone.2]
  unfold KExprMulCount
  omega

theorem constraintsFrom_totalFreshCount
    (current : KExpr)
    (rounds :
      List (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4))
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalFreshCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom
        current rounds) =
      match rounds with
      | [] => 0
      | _ :: _ => KExprMulCount current + 312 +
          (rounds.length - 1) * 467 := by
  induction rounds generalizing current with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      have headLinear := linear round (by simp)
      have tailLinear : ∀ later ∈ rounds, RoundLinear later := by
        intro later member
        exact linear later (by simp [member])
      rw [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom,
        R1CS.totalFreshCount_append,
        roundEqualities_totalFreshCount current round headLinear,
        inductionHypothesis
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round round.challenge) tailLinear]
      cases rounds with
      | nil => simp
      | cons later rest =>
          rw [evaluateRound_totalMulCount round headLinear]
          simp only [List.length_cons]
          omega

theorem constraintsFrom_totalRowCount
    (current : KExpr)
    (rounds :
      List (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4))
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalRowCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom
        current rounds) =
      match rounds with
      | [] => 0
      | _ :: _ => KExprMulCount current + 314 +
          (rounds.length - 1) * 469 := by
  induction rounds generalizing current with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      have headLinear := linear round (by simp)
      have tailLinear : ∀ later ∈ rounds, RoundLinear later := by
        intro later member
        exact linear later (by simp [member])
      rw [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom,
        R1CS.totalRowCount_append,
        roundEqualities_totalRowCount current round headLinear,
        inductionHypothesis
          (NightstreamFPrime.Gadgets.SumCheck.FixedChain.evaluateRound
            round round.challenge) tailLinear]
      cases rounds with
      | nil => simp
      | cons later rest =>
          rw [evaluateRound_totalMulCount round headLinear]
          simp only [List.length_cons]
          omega

theorem constraintsFrom_totalFreshCount_of_nonempty
    (current : KExpr)
    (rounds :
      List (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4))
    (nonempty : rounds ≠ [])
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalFreshCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom
        current rounds) =
      KExprMulCount current + 312 + (rounds.length - 1) * 467 := by
  rw [constraintsFrom_totalFreshCount current rounds linear]
  cases rounds with
  | nil => exact False.elim (nonempty rfl)
  | cons round rounds => rfl

theorem constraintsFrom_totalRowCount_of_nonempty
    (current : KExpr)
    (rounds :
      List (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round 4))
    (nonempty : rounds ≠ [])
    (linear : ∀ round ∈ rounds, RoundLinear round) :
    R1CS.totalRowCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraintsFrom
        current rounds) =
      KExprMulCount current + 314 + (rounds.length - 1) * 469 := by
  rw [constraintsFrom_totalRowCount current rounds linear]
  cases rounds with
  | nil => exact False.elim (nonempty rfl)
  | cons round rounds => rfl

end NightstreamFPrime.Layout.SumCheck.FixedChain
