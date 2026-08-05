import Nightstream.SuperNeo.InteractiveReduction.Asymptotic
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

/-!
Expected runtime of Appendix D.4's success-gated extractor.

Owns: one-run experiment and cost families, the initial-run gate, the exact
expected work expression, termination of the gated retry, and the derived
expected-polynomial-time theorem with no success floor.

Does not own: PiCCS events, extraction soundness, Fiat--Shamir, Rust, R1CS, or
constraints.

The extractor always pays for one initial run. It enters the first-success
retry only when that initial run succeeds. Therefore the retry contribution is
`p * (mean / p)`, which is `mean` for positive `p` and zero for `p = 0`.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime

open Nightstream.SuperNeo.InteractiveReduction.Asymptotic
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

universe uAdversary uSeed uOutcome

variable {Adversary : Type uAdversary}

/-- Primitive operational data for one complete execution at each security
parameter. -/
structure Family (Adversary : Type uAdversary) where
  Outcome : Nat -> Type uOutcome
  experiment : Adversary -> (securityParameter : Nat) ->
    Experiment.{uSeed, uOutcome} (Outcome securityParameter)
  success : (securityParameter : Nat) -> Outcome securityParameter -> Bool
  runCost :
    (adversary : Adversary) ->
    (securityParameter : Nat) ->
    (experiment adversary securityParameter).Seed -> Nat
  runCostBound : Adversary -> Nat -> Nat
  runCost_le_bound :
    forall adversary securityParameter seed,
      seed ∈ (experiment adversary securityParameter).support.values ->
        runCost adversary securityParameter seed <=
          runCostBound adversary securityParameter

/-- The one-run adversary cost owner is polynomially bounded. -/
def AdversaryExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary) : Prop :=
  PolynomiallyBounded
    (fun securityParameter =>
      (family.runCostBound adversary securityParameter : Rat))

private theorem sum_runCost_le
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat)
    (seeds : List (family.experiment adversary securityParameter).Seed)
    (members : forall seed, seed ∈ seeds ->
      seed ∈ (family.experiment adversary securityParameter).support.values) :
    (seeds.map (family.runCost adversary securityParameter)).sum <=
      seeds.length * family.runCostBound adversary securityParameter := by
  induction seeds with
  | nil => simp
  | cons head tail inductionHypothesis =>
      have headBound := family.runCost_le_bound adversary securityParameter
        head (members head (by simp))
      have tailBound := inductionHypothesis (by
        intro seed member
        exact members seed (by simp [member]))
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      calc
        family.runCost adversary securityParameter head +
              (tail.map (family.runCost adversary securityParameter)).sum <=
            family.runCostBound adversary securityParameter +
              tail.length * family.runCostBound adversary securityParameter :=
          Nat.add_le_add headBound tailBound
        _ = (tail.length + 1) *
              family.runCostBound adversary securityParameter := by
          rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]

/-- Exact expected cost of one complete execution. -/
def oneRunExpectedWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    (family.experiment adversary securityParameter).expectedCost
      (family.runCost adversary securityParameter)

theorem oneRunExpectedWork_nonnegative
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat) :
    0 <= oneRunExpectedWork family adversary securityParameter := by
  unfold oneRunExpectedWork Experiment.expectedCost
  rw [Rat.div_def]
  exact Rat.mul_nonneg Rat.natCast_nonneg
    (Rat.le_of_lt (Rat.inv_pos.mpr
      (Rat.natCast_pos.mpr
        (family.experiment adversary securityParameter).support.cardinality_pos)))

theorem oneRunExpectedWork_le_bound
    (family : Family Adversary)
    (adversary : Adversary) :
    forall securityParameter,
      oneRunExpectedWork family adversary securityParameter <=
        (family.runCostBound adversary securityParameter : Rat) := by
  intro securityParameter
  let experiment := family.experiment adversary securityParameter
  have sumBound := sum_runCost_le family adversary securityParameter
    experiment.support.values (fun _seed member => member)
  apply ((experiment.expectedCost_le_iff_expectedCostAtMost
    (family.runCost adversary securityParameter)
    (family.runCostBound adversary securityParameter))).mpr
  unfold Experiment.ExpectedCostAtMost Experiment.totalCost
  change
    (experiment.support.values.map
      (family.runCost adversary securityParameter)).sum <=
      family.runCostBound adversary securityParameter *
        experiment.support.values.length
  rw [Nat.mul_comm]
  exact sumBound

/-- Exact expected work of the gated retry branch, including the probability
that the branch is entered. -/
def gatedRetryExpectedWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    let probability := successProbability
      (family.experiment adversary securityParameter)
      (family.success securityParameter)
    probability *
      (oneRunExpectedWork family adversary securityParameter *
        (1 / probability))

/-- Total expected work: one initial run plus the gated retry. -/
def expectedWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    oneRunExpectedWork family adversary securityParameter +
      gatedRetryExpectedWork family adversary securityParameter

/-- The gated retry costs at most one additional one-run expectation. -/
theorem gatedRetryExpectedWork_le_oneRun
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat) :
    gatedRetryExpectedWork family adversary securityParameter <=
      oneRunExpectedWork family adversary securityParameter := by
  let probability := successProbability
    (family.experiment adversary securityParameter)
    (family.success securityParameter)
  let mean := oneRunExpectedWork family adversary securityParameter
  by_cases probabilityZero : probability = 0
  · change probability * (mean * (1 / probability)) <= mean
    rw [probabilityZero]
    simp only [Rat.zero_mul]
    exact oneRunExpectedWork_nonnegative family adversary securityParameter
  · have cancels : probability * (1 / probability) = 1 := by
      rw [Rat.div_def, Rat.one_mul]
      exact Rat.mul_inv_cancel probability probabilityZero
    have exactWork : probability * (mean * (1 / probability)) = mean := by
      calc
        probability * (mean * (1 / probability)) =
            (probability * mean) * (1 / probability) :=
          (Rat.mul_assoc _ _ _).symm
        _ = (mean * probability) * (1 / probability) := by
          rw [Rat.mul_comm probability mean]
        _ = mean * (probability * (1 / probability)) :=
          Rat.mul_assoc _ _ _
        _ = mean := by rw [cancels, Rat.mul_one]
    change probability * (mean * (1 / probability)) <= mean
    rw [exactWork]
    exact Rat.le_refl

/-- Pointwise two-run work bound. -/
def twoRunWorkBound
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    (family.runCostBound adversary securityParameter : Rat) +
      (family.runCostBound adversary securityParameter : Rat)

theorem expectedWork_le_twoRunWorkBound
    (family : Family Adversary)
    (adversary : Adversary) :
    forall securityParameter,
      expectedWork family adversary securityParameter <=
        twoRunWorkBound family adversary securityParameter := by
  intro securityParameter
  unfold expectedWork twoRunWorkBound
  exact Rat.le_trans
    ((Rat.add_le_add_left
      (c := oneRunExpectedWork family adversary securityParameter)).mpr
      (gatedRetryExpectedWork_le_oneRun family adversary securityParameter))
    (Rat.le_trans
      ((Rat.add_le_add_right
        (c := oneRunExpectedWork family adversary securityParameter)).mpr
        (oneRunExpectedWork_le_bound family adversary securityParameter))
      ((Rat.add_le_add_left
        (c := (family.runCostBound adversary securityParameter : Rat))).mpr
        (oneRunExpectedWork_le_bound family adversary securityParameter)))

/-- Termination of the complete gated algorithm at one parameter. If success
probability is zero, the retry branch is never entered. Otherwise the retry
terminates almost surely. -/
def TerminatesAt
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat) : Prop :=
  successProbability
      (family.experiment adversary securityParameter)
      (family.success securityParameter) = 0 \/
    AlmostSurelyTerminates
      (family.experiment adversary securityParameter)
      (family.success securityParameter)

theorem terminatesAt
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat) :
    TerminatesAt family adversary securityParameter := by
  let experiment := family.experiment adversary securityParameter
  let success := family.success securityParameter
  by_cases nonempty :
      experiment.support.values.filter
        (fun seed => success (experiment.outcome seed)) ≠ []
  · exact Or.inr (firstSuccess_terminates_almostSurely
      experiment success nonempty)
  · have filteredEmpty :
        experiment.support.values.filter
          (fun seed => success (experiment.outcome seed)) = [] :=
      Classical.not_not.mp nonempty
    have countZero : experiment.countBool success = 0 := by
      unfold Experiment.countBool
      rw [List.countP_eq_length_filter, filteredEmpty]
      rfl
    apply Or.inl
    unfold successProbability Experiment.probabilityBool
    rw [countZero]
    simp [Rat.div_def]

/-- Expected-polynomial-time contract for the complete success-gated
extractor. -/
def ExtractorExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary) : Prop :=
  (forall securityParameter, TerminatesAt family adversary securityParameter) /\
    PolynomiallyBounded (expectedWork family adversary)

theorem extractorExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary)
    (adversaryEpt : AdversaryExpectedPolynomialTime family adversary) :
    ExtractorExpectedPolynomialTime family adversary := by
  constructor
  · exact terminatesAt family adversary
  · exact PolynomiallyBounded.mono
      (expectedWork_le_twoRunWorkBound family adversary)
      (adversaryEpt.add adversaryEpt)

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime
