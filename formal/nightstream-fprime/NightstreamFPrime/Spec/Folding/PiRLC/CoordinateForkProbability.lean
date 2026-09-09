import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetry
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Logic.Equiv.Prod

/-!
Combines the interactive coordinate-retry experiment over a uniform challenge
vector. Each response may use fresh random coins. The base is queried once;
after acceptance, each coordinate retries until acceptance, and a repeated
coordinate is failure. All other coordinates stay at the base values.

The quantities below are the probabilities and expected calls of that experiment,
derived from its per-line first-hit laws. They do not assert Fiat--Shamir security
or apply a uniform law to the production sampler.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkProbability

open scoped BigOperators
open CoordinateRetry

variable {Index Challenge : Type*} [Fintype Index] [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge]

private theorem rate_eq_expect {Domain : Type*} [Fintype Domain]
    (line : Line Domain) : line.rate = 𝔼 value, line.acceptance value := by
  simp only [Fintype.expect_eq_sum_div_card, Line.rate, Line.weight, ← Finset.sum_div]

omit [Nonempty Challenge] in
private theorem repeated_eq_expect (line : Line Challenge) :
    line.repeatedBaseMass =
      𝔼 challenge, line.acceptance challenge *
        (line.weight challenge / line.rate) := by
  rw [Fintype.expect_eq_sum_div_card, Finset.sum_div, Line.repeatedBaseMass,
    Finset.sum_div]
  apply Finset.sum_congr rfl
  intro challenge _
  unfold Line.weight
  ring

private theorem repeated_le_inverse (line : Line Challenge) :
    line.repeatedBaseMass ≤ 1 / Fintype.card Challenge := by
  by_cases zero : line.rate = 0
  · simp only [Line.repeatedBaseMass, zero, div_zero]
    positivity
  · exact line.repeatedBaseMass_le_inverse
      (lt_of_le_of_ne line.rate_nonnegative (Ne.symm zero))

omit [Nonempty Challenge] in
private theorem average_conditional_calls_le_one (line : Line Challenge) :
    (𝔼 challenge, line.acceptance challenge / line.rate) ≤ 1 := by
  rw [← Finset.expect_div, ← rate_eq_expect]
  by_cases zero : line.rate = 0
  · simp [zero]
  · simp [div_self zero]

/-- Acceptance oracle on one coordinate, holding every other coordinate fixed. -/
noncomputable def coordinateLine (base : Line (Index → Challenge))
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    Line Challenge where
  acceptance challenge := base.acceptance
    ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest))
  nonnegative challenge := base.nonnegative
    ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest))
  atMostOne challenge := base.atMostOne
    ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest))

/-- Conditional chance that the first accepted coordinate retry repeats the
base challenge. A zero-rate line is never entered after base acceptance. -/
noncomputable def repeatChance (base : Line (Index → Challenge))
    (vector : Index → Challenge) (coordinate : Index) : ℝ :=
  let parts := Equiv.funSplitAt coordinate Challenge vector
  let line := coordinateLine base coordinate parts.2
  line.weight parts.1 / line.rate

/-- Conditional expected coordinate calls, with the zero-rate branch unused. -/
noncomputable def conditionalCalls (base : Line (Index → Challenge))
    (vector : Index → Challenge) (coordinate : Index) : ℝ :=
  let parts := Equiv.funSplitAt coordinate Challenge vector
  1 / (coordinateLine base coordinate parts.2).rate

omit [Nonempty Challenge] in
private theorem split_average (coordinate : Index) (function : (Index → Challenge) → ℝ) :
    (𝔼 vector, function vector) =
      𝔼 rest, 𝔼 challenge,
        function ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) := by
  let split := Equiv.funSplitAt coordinate Challenge
  calc
    (𝔼 vector, function vector) = 𝔼 pair, function (split.symm pair) :=
      Fintype.expect_equiv split function (fun pair => function (split.symm pair))
        (by intro vector; simp)
    _ = 𝔼 challenge, 𝔼 rest, function (split.symm (challenge, rest)) := by
      rw [← Finset.univ_product_univ, Finset.expect_product]
    _ = 𝔼 rest, 𝔼 challenge, function (split.symm (challenge, rest)) :=
      Finset.expect_comm _ _ _

omit [Fintype Index] in
theorem repeatChance_nonnegative (base : Line (Index → Challenge))
    (vector : Index → Challenge) (coordinate : Index) :
    0 ≤ repeatChance base vector coordinate := by
  unfold repeatChance
  exact div_nonneg (Line.weight_nonnegative _ _) (Line.rate_nonnegative _)

omit [Fintype Index] in
theorem repeatChance_le_one (base : Line (Index → Challenge))
    (vector : Index → Challenge) (coordinate : Index) :
    repeatChance base vector coordinate ≤ 1 := by
  unfold repeatChance
  dsimp only
  let line := coordinateLine base coordinate
    (Equiv.funSplitAt coordinate Challenge vector).2
  change line.weight _ / line.rate ≤ 1
  by_cases zero : line.rate = 0
  · simp [zero]
  · exact (div_le_one (lt_of_le_of_ne line.rate_nonnegative (Ne.symm zero))).mpr
      (line.weight_le_rate _)

/-- Averaging over the other coordinates preserves the exact `1 / |C|` loss. -/
theorem coordinate_repeat_loss (base : Line (Index → Challenge))
    (coordinate : Index) :
    (𝔼 vector, base.acceptance vector * repeatChance base vector coordinate) ≤
      1 / Fintype.card Challenge := by
  rw [split_average coordinate]
  apply Finset.expect_le Finset.univ_nonempty
  intro rest _
  have equation :
      (𝔼 challenge,
        base.acceptance ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) *
          repeatChance base
            ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) coordinate) =
        (coordinateLine base coordinate rest).repeatedBaseMass := by
    rw [repeated_eq_expect]
    apply Finset.expect_congr rfl
    intro challenge _
    simp only [repeatChance, Equiv.apply_symm_apply]
    rfl
  rw [equation]
  exact repeated_le_inverse _

/-- The expected number of retry invocations for one coordinate is at most
one. Initial rejection is included through `base.acceptance`. -/
theorem coordinate_expected_calls (base : Line (Index → Challenge))
    (coordinate : Index) :
    (𝔼 vector, base.acceptance vector * conditionalCalls base vector coordinate) ≤ 1 := by
  rw [split_average coordinate]
  apply Finset.expect_le Finset.univ_nonempty
  intro rest _
  have equation :
      (𝔼 challenge,
        base.acceptance ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) *
          conditionalCalls base
            ((Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)) coordinate) =
        𝔼 challenge, (coordinateLine base coordinate rest).acceptance challenge /
          (coordinateLine base coordinate rest).rate := by
    apply Finset.expect_congr rfl
    intro challenge _
    simp only [conditionalCalls, Equiv.apply_symm_apply, coordinateLine,
      div_eq_mul_inv, one_mul]
  rw [equation]
  exact average_conditional_calls_le_one _

/-- Probability of an accepted base and distinct first accepted retries at
every coordinate. The searches use independent oracle coins given the base. -/
noncomputable def successProbability (base : Line (Index → Challenge)) : ℝ :=
  𝔼 vector, base.acceptance vector *
    ∏ coordinate, (1 - repeatChance base vector coordinate)

/-- One initial call, then every coordinate search after initial acceptance. -/
noncomputable def expectedOracleCalls (base : Line (Index → Challenge)) : ℝ :=
  1 + ∑ coordinate,
    𝔼 vector, base.acceptance vector * conditionalCalls base vector coordinate

omit [Fintype Index] in
private theorem product_lower_bound (probability : Index → ℝ)
    (nonnegative : ∀ index, 0 ≤ probability index)
    (atMostOne : ∀ index, probability index ≤ 1) (indices : Finset Index) :
    1 - ∑ index ∈ indices, probability index ≤
      ∏ index ∈ indices, (1 - probability index) := by
  induction indices using Finset.induction_on with
  | empty => simp
  | @insert index indices outside inductionHypothesis =>
      rw [Finset.sum_insert outside, Finset.prod_insert outside]
      have productBound : (∏ item ∈ indices, (1 - probability item)) ≤ 1 :=
        Finset.prod_le_one
          (fun item _ => sub_nonneg.mpr (atMostOne item))
          (fun item _ => sub_le_self 1 (nonnegative item))
      have currentNonnegative := nonnegative index
      nlinarith

/-- The interactive extractor loses at most one inverse challenge-set size
per coordinate. Rejected oracle responses are included in the first-hit law. -/
theorem successProbability_lower_bound (base : Line (Index → Challenge)) :
    base.rate - (Fintype.card Index : ℝ) / Fintype.card Challenge ≤
      successProbability base := by
  have lower :
      (𝔼 vector, (base.acceptance vector -
        ∑ coordinate, base.acceptance vector * repeatChance base vector coordinate)) ≤
      successProbability base := by
    apply Finset.expect_le_expect
    intro vector _
    have pointwise := mul_le_mul_of_nonneg_left
      (product_lower_bound (repeatChance base vector)
        (repeatChance_nonnegative base vector) (repeatChance_le_one base vector)
        Finset.univ) (base.nonnegative vector)
    simpa only [mul_sub, mul_one, Finset.mul_sum] using pointwise
  have loss :
      (∑ coordinate, 𝔼 vector,
        base.acceptance vector * repeatChance base vector coordinate) ≤
        (Fintype.card Index : ℝ) / Fintype.card Challenge := by
    calc
      _ ≤ ∑ _ : Index, (1 : ℝ) / Fintype.card Challenge :=
        Finset.sum_le_sum fun coordinate _ => coordinate_repeat_loss base coordinate
      _ = (Fintype.card Index : ℝ) / Fintype.card Challenge := by
        simp [div_eq_mul_inv]
  rw [Finset.expect_sub_distrib, Finset.expect_sum_comm, ← rate_eq_expect] at lower
  linarith

/-- The oracle-call bound includes failed retries, weighted by the chance that
the base was accepted. It is not a bound conditioned on base acceptance. -/
theorem expectedOracleCalls_le (base : Line (Index → Challenge)) :
    expectedOracleCalls base ≤ (Fintype.card Index : ℝ) + 1 := by
  have bound :
      (∑ coordinate, 𝔼 vector,
        base.acceptance vector * conditionalCalls base vector coordinate) ≤
        Fintype.card Index := by
    calc
      _ ≤ ∑ _ : Index, (1 : ℝ) :=
        Finset.sum_le_sum fun coordinate _ => coordinate_expected_calls base coordinate
      _ = Fintype.card Index := by simp
  unfold expectedOracleCalls
  linarith

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkProbability
