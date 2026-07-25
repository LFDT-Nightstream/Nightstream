import Init.Omega
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

/-!
Security-parameter-indexed runtime theorem for the unbounded finite-alphabet
first-success sampler.

Owns: one-run experiment/cost families, verifier-call cost ownership, the
derived pathwise trace-work bound, a concrete adversary polynomial-time
predicate, exact geometric expected work, and derivation of almost-sure
termination plus expected polynomial time from a positive inverse-polynomial
success floor.

Does not own: PiCCS events, a protocol security contract, Fiat--Shamir, Rust,
R1CS, or constraints.

Neither termination nor extractor EPT is a field of `Family`. They are
theorems derived from its finite one-run experiment, explicit costs, and
success-floor arithmetic.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime

open Nightstream.SuperNeo.InteractiveReduction.Asymptotic
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

universe uAdversary uSeed uOutcome

variable {Adversary : Type uAdversary}

/-- Primitive operational data for one complete execution at each security
parameter. The seed type remains owned existentially by `Experiment`. -/
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
  successFloor : Weight
  successFloor_pos :
    forall securityParameter, 0 < successFloor securityParameter
  inverseFloorPolynomial :
    PolynomiallyBounded
      (fun securityParameter => 1 / successFloor securityParameter)

/-- The adversary is polynomial time exactly when the explicit pointwise
one-run cost bound is polynomially bounded. -/
def AdversaryExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary) : Prop :=
  PolynomiallyBounded
    (fun securityParameter =>
      (family.runCostBound adversary securityParameter : Rat))

/-- Extraction is eligible exactly when the declared positive floor bounds
the actual one-run success probability at every security parameter. -/
def ExtractionEligible
    (family : Family Adversary)
    (adversary : Adversary) : Prop :=
  forall securityParameter,
    family.successFloor securityParameter <=
      successProbability
        (family.experiment adversary securityParameter)
        (family.success securityParameter)

private theorem successfulSupport_nonempty
    (family : Family Adversary)
    (adversary : Adversary)
    (eligible : ExtractionEligible family adversary)
    (securityParameter : Nat) :
    (family.experiment adversary securityParameter).support.values.filter
      (fun seed =>
        family.success securityParameter
          ((family.experiment adversary securityParameter).outcome seed)) ≠
      [] := by
  let experiment := family.experiment adversary securityParameter
  let success := family.success securityParameter
  have probabilityPos :
      0 < experiment.probabilityBool success := by
    apply Rat.not_le.mp
    intro probabilityNonpositive
    exact (Rat.not_le.mpr
      (family.successFloor_pos securityParameter))
      (Rat.le_trans
        (eligible securityParameter) probabilityNonpositive)
  intro empty
  have countZero : experiment.countBool success = 0 := by
    unfold Experiment.countBool
    rw [List.countP_eq_length_filter, empty]
    rfl
  unfold Experiment.probabilityBool at probabilityPos
  rw [countZero] at probabilityPos
  simp [Rat.div_def] at probabilityPos

/-- Number of complete executions owned by one terminating trace, including
the independently fresh second execution. -/
def Trace.executionCount
    {Outcome : Type uOutcome}
    {experiment : Experiment Outcome}
    {success : Outcome -> Bool}
    (trace : Trace experiment success) : Nat :=
  trace.failed.length + 2

/-- Concrete work of a terminating trace. -/
def traceWork
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat)
    (trace : Trace
      (family.experiment adversary securityParameter)
      (family.success securityParameter)) : Nat :=
  (trace.failed.map
      (family.runCost adversary securityParameter)).sum +
    family.runCost adversary securityParameter trace.first +
    family.runCost adversary securityParameter trace.fresh

private theorem sum_runCost_le
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat)
    (seeds : List
      (family.experiment adversary securityParameter).Seed)
    (members : forall seed, seed ∈ seeds ->
      seed ∈
        (family.experiment adversary securityParameter).support.values) :
    (seeds.map
      (family.runCost adversary securityParameter)).sum <=
      seeds.length * family.runCostBound adversary securityParameter := by
  induction seeds with
  | nil => simp
  | cons head tail inductionHypothesis =>
      have headBound :=
        family.runCost_le_bound adversary securityParameter head
          (members head (by simp))
      have tailMembers : forall seed, seed ∈ tail ->
          seed ∈
            (family.experiment adversary securityParameter).support.values := by
        intro seed member
        exact members seed (by simp [member])
      have tailBound := inductionHypothesis tailMembers
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      calc
        family.runCost adversary securityParameter head +
              (tail.map
                (family.runCost adversary securityParameter)).sum <=
            family.runCostBound adversary securityParameter +
              tail.length *
                family.runCostBound adversary securityParameter :=
          Nat.add_le_add headBound tailBound
        _ = (tail.length + 1) *
              family.runCostBound adversary securityParameter := by
          rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]

/-- Every terminating trace costs at most its execution count times the
explicit one-run bound. -/
theorem traceWork_le_executionCount_mul_bound
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat)
    (trace : Trace
      (family.experiment adversary securityParameter)
      (family.success securityParameter)) :
    traceWork family adversary securityParameter trace <=
      Trace.executionCount trace *
        family.runCostBound adversary securityParameter := by
  have failedBound :=
    sum_runCost_le family adversary securityParameter
      trace.failed trace.failed_mem
  have firstBound :=
    family.runCost_le_bound adversary securityParameter trace.first
      trace.first_mem
  have freshBound :=
    family.runCost_le_bound adversary securityParameter trace.fresh
      trace.fresh_mem
  unfold traceWork Trace.executionCount
  calc
    (trace.failed.map
          (family.runCost adversary securityParameter)).sum +
          family.runCost adversary securityParameter trace.first +
        family.runCost adversary securityParameter trace.fresh <=
      trace.failed.length *
            family.runCostBound adversary securityParameter +
          family.runCostBound adversary securityParameter +
        family.runCostBound adversary securityParameter :=
      Nat.add_le_add
        (Nat.add_le_add failedBound firstBound) freshBound
    _ = (trace.failed.length + 2) *
        family.runCostBound adversary securityParameter := by
      rw [Nat.add_mul, Nat.two_mul, Nat.add_assoc]

/-- Exact expected cost of one complete execution under the one-run seed
law. -/
def oneRunExpectedWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    (family.experiment adversary securityParameter).expectedCost
      (family.runCost adversary securityParameter)

/-- The one-run expected cost is bounded by the pointwise run-cost owner. -/
theorem oneRunExpectedWork_le_bound
    (family : Family Adversary)
    (adversary : Adversary) :
    forall securityParameter,
      oneRunExpectedWork family adversary securityParameter <=
        (family.runCostBound adversary securityParameter : Rat) := by
  intro securityParameter
  let experiment := family.experiment adversary securityParameter
  have sumBound :=
    sum_runCost_le family adversary securityParameter
      experiment.support.values
      (fun _seed member => member)
  apply
    ((experiment.expectedCost_le_iff_expectedCostAtMost
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

/-- Exact expected retry work: one-run expected cost times the exact
geometric expected retry count. -/
def expectedRetryWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    oneRunExpectedWork family adversary securityParameter *
      expectedRetryExecutions
        (family.experiment adversary securityParameter)
        (family.success securityParameter)

/-- Exact expected total work, including one independently fresh execution
after the first successful execution. -/
def expectedWork
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    expectedRetryWork family adversary securityParameter +
      oneRunExpectedWork family adversary securityParameter

/-- The retry-work expectation satisfies the operational first-step equation:
pay one complete execution now, then restart only on failure. -/
theorem expectedRetryWork_firstStep
    (family : Family Adversary)
    (adversary : Adversary)
    (eligible : ExtractionEligible family adversary)
    (securityParameter : Nat) :
    expectedRetryWork family adversary securityParameter =
      oneRunExpectedWork family adversary securityParameter +
        failureProbability
            (family.experiment adversary securityParameter)
            (family.success securityParameter) *
          expectedRetryWork family adversary securityParameter := by
  let experiment := family.experiment adversary securityParameter
  let success := family.success securityParameter
  let mean := oneRunExpectedWork family adversary securityParameter
  let retries := expectedRetryExecutions experiment success
  let failure := failureProbability experiment success
  have nonempty :=
    successfulSupport_nonempty
      family adversary eligible securityParameter
  have recurrence :
      retries = 1 + failure * retries :=
    expectedRetryExecutions_firstStep experiment success nonempty
  change mean * retries = mean + failure * (mean * retries)
  calc
    mean * retries =
        mean * (1 + failure * retries) :=
      congrArg (fun value : Rat => mean * value) recurrence
    _ = mean + mean * (failure * retries) := by
      rw [Rat.mul_add, Rat.mul_one]
    _ = mean + (mean * failure) * retries := by
      rw [Rat.mul_assoc]
    _ = mean + (failure * mean) * retries := by
      rw [Rat.mul_comm mean failure]
    _ = mean + failure * (mean * retries) := by
      rw [Rat.mul_assoc]

/-- Expected total work is the exact one-run mean multiplied by the exact
expected execution count `1 / p + 1`. -/
theorem expectedWork_eq_oneRun_mul_totalExecutions
    (family : Family Adversary)
    (adversary : Adversary)
    (securityParameter : Nat) :
    expectedWork family adversary securityParameter =
      oneRunExpectedWork family adversary securityParameter *
        expectedTotalExecutions
          (family.experiment adversary securityParameter)
          (family.success securityParameter) := by
  unfold expectedWork expectedRetryWork expectedTotalExecutions
  rw [Rat.mul_add, Rat.mul_one]

/-- The floor-facing work bound used to prove polynomiality. -/
def floorWorkBound
    (family : Family Adversary)
    (adversary : Adversary) : Weight :=
  fun securityParameter =>
    (family.runCostBound adversary securityParameter : Rat) *
      (1 / family.successFloor securityParameter + 1)

/-- The actual geometric expected-work bound is no larger than the
inverse-floor bound. -/
theorem expectedWork_le_floorWorkBound
    (family : Family Adversary)
    (adversary : Adversary)
    (eligible : ExtractionEligible family adversary) :
    forall securityParameter,
      expectedWork family adversary securityParameter <=
        floorWorkBound family adversary securityParameter := by
  intro securityParameter
  let experiment := family.experiment adversary securityParameter
  let success := family.success securityParameter
  have inverseBound :
      1 / successProbability experiment success <=
        1 / family.successFloor securityParameter := by
    exact div_le_div_of_nonneg_of_le_of_pos_le
      (by decide : (0 : Rat) <= 1) Rat.le_refl
      (family.successFloor_pos securityParameter)
      (eligible securityParameter)
  have executionBound :
      expectedTotalExecutions experiment success <=
        1 / family.successFloor securityParameter + 1 := by
    unfold expectedTotalExecutions expectedRetryExecutions
    exact (Rat.add_le_add_right (c := 1)).mpr inverseBound
  have executionNonnegative :
      0 <= expectedTotalExecutions experiment success := by
    unfold expectedTotalExecutions expectedRetryExecutions
    have inversePositive :
        0 < 1 / successProbability experiment success := by
      rw [Rat.div_def, Rat.one_mul]
      exact Rat.inv_pos.mpr
        (successProbability_pos experiment success
          (successfulSupport_nonempty
            family adversary eligible securityParameter))
    exact Rat.add_nonneg (Rat.le_of_lt inversePositive)
      (by decide : (0 : Rat) <= 1)
  rw [expectedWork_eq_oneRun_mul_totalExecutions]
  unfold floorWorkBound
  calc
    oneRunExpectedWork family adversary securityParameter *
          expectedTotalExecutions experiment success <=
        (family.runCostBound adversary securityParameter : Rat) *
          expectedTotalExecutions experiment success :=
      Rat.mul_le_mul_of_nonneg_right
        (oneRunExpectedWork_le_bound
          family adversary securityParameter)
        executionNonnegative
    _ <=
        (family.runCostBound adversary securityParameter : Rat) *
          (1 / family.successFloor securityParameter + 1) :=
      Rat.mul_le_mul_of_nonneg_left executionBound Rat.natCast_nonneg

private theorem inverseFloor_add_one_polynomial
    (family : Family Adversary) :
    PolynomiallyBounded
      (fun securityParameter =>
        1 / family.successFloor securityParameter + 1) :=
  family.inverseFloorPolynomial.add polynomiallyBounded_one

private theorem floorWorkBound_polynomial
    (family : Family Adversary)
    (adversary : Adversary)
    (adversaryEpt : AdversaryExpectedPolynomialTime family adversary) :
    PolynomiallyBounded (floorWorkBound family adversary) := by
  apply PolynomiallyBounded.mul_of_nonnegative
  · intro securityParameter
    have inversePositive :
        0 < 1 / family.successFloor securityParameter := by
      rw [Rat.div_def, Rat.one_mul]
      exact Rat.inv_pos.mpr
        (family.successFloor_pos securityParameter)
    exact Rat.add_nonneg (Rat.le_of_lt inversePositive)
      (by decide : (0 : Rat) <= 1)
  · exact adversaryEpt
  · exact inverseFloor_add_one_polynomial family

/-- Extractor EPT is a derived conjunction: every parameter's unbounded
sampler terminates almost surely, and its exact geometric work bound is
polynomial. -/
def ExtractorExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary) : Prop :=
  (forall securityParameter,
    AlmostSurelyTerminates
      (family.experiment adversary securityParameter)
      (family.success securityParameter)) /\
  PolynomiallyBounded (expectedWork family adversary)

/-- Positive inverse-polynomial success and polynomial one-run work derive
the unbounded sampler's almost-sure termination and expected polynomial
time. -/
theorem extractorExpectedPolynomialTime
    (family : Family Adversary)
    (adversary : Adversary)
    (adversaryEpt : AdversaryExpectedPolynomialTime family adversary)
    (eligible : ExtractionEligible family adversary) :
    ExtractorExpectedPolynomialTime family adversary := by
  constructor
  · intro securityParameter
    exact firstSuccess_terminates_almostSurely
      (family.experiment adversary securityParameter)
      (family.success securityParameter)
      (successfulSupport_nonempty
        family adversary eligible securityParameter)
  · exact PolynomiallyBounded.mono
      (expectedWork_le_floorWorkBound family adversary eligible)
      (floorWorkBound_polynomial family adversary adversaryEpt)

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime
