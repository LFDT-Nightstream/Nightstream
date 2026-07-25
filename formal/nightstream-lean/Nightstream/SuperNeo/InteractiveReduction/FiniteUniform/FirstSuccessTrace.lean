import Nightstream.SuperNeo.InteractiveReduction.Asymptotic
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessExtraction
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Reindex
import Init.Omega

/-!
Specialized countable first-success trace law for a finite one-run
experiment.

Owns: terminating trace data, exact geometric length mass and failure tails,
finite-prefix mass conservation, normalization of the unbounded terminating
law, its recursive sampler equation, exact equality of the selected-first
and fresh-second output law with the finite conditioned product, and the
geometric expected execution count.

Does not own: a protocol, a security-parameter family, a concrete runtime
cost, Fiat--Shamir, Rust, R1CS, or constraints.

The unbounded law is not an arbitrary probability premise. Its event
probability is the unique finite solution of the first-step recursion

`P(A) = Pr[first succeeds and A] + Pr[first fails] * P(A)`.

Positive one-run success makes the coefficient invertible. The resulting
law is then proved equal to “successful first seed × fresh unconditioned
second seed”, so the fresh execution is never conditioned.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uSeed uOutcome

variable {Seed : Type uSeed}
variable {Outcome : Type uOutcome}

private theorem rat_lt_of_lt_of_le
    {left middle right : Rat}
    (leftMiddle : left < middle)
    (middleRight : middle <= right) :
    left < right := by
  apply Rat.lt_of_le_of_ne
    (Rat.le_trans (Rat.le_of_lt leftMiddle) middleRight)
  intro equal
  subst right
  exact (Rat.not_lt.mpr middleRight) leftMiddle

private theorem rat_lt_of_le_of_lt
    {left middle right : Rat}
    (leftMiddle : left <= middle)
    (middleRight : middle < right) :
    left < right := by
  apply Rat.lt_of_le_of_ne
    (Rat.le_trans leftMiddle (Rat.le_of_lt middleRight))
  intro equal
  subst right
  exact (Rat.not_lt.mpr leftMiddle) middleRight

/-- One terminating execution trace: zero or more failed iid seeds, the first
successful seed, and one independently fresh seed. Membership fields ensure
that no value outside the one-run support receives trace mass. -/
structure Trace
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) where
  failed : List experiment.Seed
  failed_mem : forall seed, seed ∈ failed ->
    seed ∈ experiment.support.values
  failed_reject : forall seed, seed ∈ failed ->
    success (experiment.outcome seed) = false
  first : experiment.Seed
  first_mem : first ∈ experiment.support.values
  first_accept : success (experiment.outcome first) = true
  fresh : experiment.Seed
  fresh_mem : fresh ∈ experiment.support.values

/-- Exact one-run success probability. -/
def successProbability
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Rat :=
  experiment.probabilityBool success

/-- Exact one-run failure probability. -/
def failureProbability
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Rat :=
  experiment.probabilityBool fun outcome => !success outcome

private theorem length_eq_success_add_failure
    (values : List Seed)
    (success : Seed -> Bool) :
    values.length =
      values.countP success +
        values.countP (fun seed => !success seed) := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases headSuccess : success head with
      | false =>
          simp [headSuccess, inductionHypothesis]
          omega
      | true =>
          simp [headSuccess, inductionHypothesis]
          omega

/-- Success and failure partition the complete one-run support. -/
theorem success_add_failure
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) :
    successProbability experiment success +
        failureProbability experiment success = 1 := by
  let seedSuccess : experiment.Seed -> Bool :=
    fun seed => success (experiment.outcome seed)
  have partition :
      experiment.support.values.length =
        experiment.support.values.countP seedSuccess +
          experiment.support.values.countP
            (fun seed => !seedSuccess seed) :=
    length_eq_success_add_failure experiment.support.values seedSuccess
  have cardinalityNe :
      (experiment.support.cardinality : Rat) ≠ 0 :=
    Rat.ne_of_gt (Rat.natCast_pos.mpr
      experiment.support.cardinality_pos)
  unfold successProbability failureProbability
    Experiment.probabilityBool Experiment.countBool
  change
    (experiment.support.values.countP seedSuccess : Rat) /
          (experiment.support.cardinality : Rat) +
        (experiment.support.values.countP
          (fun seed => !seedSuccess seed) : Rat) /
          (experiment.support.cardinality : Rat) = 1
  simp only [Rat.div_def]
  rw [← Rat.add_mul, ← Rat.natCast_add, ← partition]
  change
    (experiment.support.cardinality : Rat) *
        (experiment.support.cardinality : Rat)⁻¹ = 1
  exact Rat.mul_inv_cancel _ cardinalityNe

/-- Failure is exactly one minus success. -/
theorem failure_eq_one_sub_success
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) :
    failureProbability experiment success =
      1 - successProbability experiment success := by
  have partition := success_add_failure experiment success
  calc
    failureProbability experiment success =
        (failureProbability experiment success +
            successProbability experiment success) -
          successProbability experiment success :=
      (Rat.add_sub_cancel).symm
    _ =
        (successProbability experiment success +
            failureProbability experiment success) -
          successProbability experiment success := by
      rw [Rat.add_comm]
    _ = 1 - successProbability experiment success := by
      rw [partition]

/-- Positive success support gives a positive success probability. -/
theorem successProbability_pos
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    0 < successProbability experiment success :=
  experiment.probabilityBool_pos_of_filter_nonempty success nonempty

/-- Geometric mass of traces with exactly `failedCount` failed runs. The
fresh run is already marginalized and therefore contributes factor one. -/
def terminationMassAt
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (failedCount : Nat) : Rat :=
  failureProbability experiment success ^ failedCount *
    successProbability experiment success

/-- Residual probability that the first `attemptCount` runs all fail. -/
def failureTail
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptCount : Nat) : Rat :=
  failureProbability experiment success ^ attemptCount

/-- The failure tail is the exact geometric power. -/
theorem failureTail_eq_pow
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptCount : Nat) :
    failureTail experiment success attemptCount =
      failureProbability experiment success ^ attemptCount :=
  rfl

/-- Terminating mass accumulated through all trace lengths below the supplied
cutoff. -/
def partialTerminationMass
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Nat -> Rat
  | 0 => 0
  | failedCount + 1 =>
      partialTerminationMass experiment success failedCount +
        terminationMassAt experiment success failedCount

/-- At every finite cutoff, accumulated terminating mass plus the exact
all-failure tail is one. -/
theorem partialTerminationMass_add_failureTail
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) :
    forall attemptCount,
      partialTerminationMass experiment success attemptCount +
          failureTail experiment success attemptCount = 1 := by
  intro attemptCount
  induction attemptCount with
  | zero =>
      simp only [partialTerminationMass, failureTail, Rat.pow_zero,
        Rat.zero_add]
  | succ smaller inductionHypothesis =>
      let successMass := successProbability experiment success
      let failureMass := failureProbability experiment success
      have partition :
          successMass + failureMass = 1 :=
        success_add_failure experiment success
      change
        (partialTerminationMass experiment success smaller +
            failureMass ^ smaller * successMass) +
          failureMass ^ (smaller + 1) = 1
      rw [Rat.pow_succ]
      calc
        (partialTerminationMass experiment success smaller +
              failureMass ^ smaller * successMass) +
            failureMass ^ smaller * failureMass =
          partialTerminationMass experiment success smaller +
            failureMass ^ smaller * (successMass + failureMass) := by
              rw [Rat.mul_add, Rat.add_assoc]
        _ =
          partialTerminationMass experiment success smaller +
            failureMass ^ smaller := by
              rw [partition, Rat.mul_one]
        _ = 1 := inductionHypothesis

private theorem geometricTail_mul_linear_le_one
    (successMass failureMass : Rat)
    (successNonnegative : 0 <= successMass)
    (failureNonnegative : 0 <= failureMass)
    (partition : successMass + failureMass = 1) :
    forall attemptCount : Nat,
      failureMass ^ attemptCount *
          (1 + (attemptCount : Rat) * successMass) <= 1 := by
  intro attemptCount
  induction attemptCount with
  | zero =>
      simp only [Rat.pow_zero, Rat.natCast_ofNat, Rat.zero_mul,
        Rat.add_zero, Rat.one_mul]
      exact Rat.le_refl
  | succ smaller inductionHypothesis =>
      let linear : Rat := 1 + (smaller : Rat) * successMass
      have failureLeOne : failureMass <= 1 := by
        calc
          failureMass = 0 + failureMass := (Rat.zero_add _).symm
          _ <= successMass + failureMass :=
            (Rat.add_le_add_right (c := failureMass)).mpr
              successNonnegative
          _ = 1 := partition
      have oneLeLinear : 1 <= linear := by
        unfold linear
        change 1 <= 1 + (smaller : Rat) * successMass
        simpa only [Rat.add_zero] using
          ((Rat.add_le_add_left (c := 1)).mpr
            (Rat.mul_nonneg Rat.natCast_nonneg successNonnegative))
      have failureLeLinear : failureMass <= linear :=
        Rat.le_trans failureLeOne oneLeLinear
      have cross :
          failureMass * successMass <= linear * successMass :=
        Rat.mul_le_mul_of_nonneg_right failureLeLinear successNonnegative
      have branchBound :
          failureMass * (linear + successMass) <= linear := by
        calc
          failureMass * (linear + successMass) =
              failureMass * linear + failureMass * successMass := by
            rw [Rat.mul_add]
          _ <= failureMass * linear + linear * successMass :=
            (Rat.add_le_add_left
              (c := failureMass * linear)).mpr cross
          _ = failureMass * linear + successMass * linear :=
            congrArg (fun value : Rat =>
              failureMass * linear + value)
              (Rat.mul_comm linear successMass)
          _ = (failureMass + successMass) * linear :=
            (Rat.add_mul failureMass successMass linear).symm
          _ = linear := by
            rw [Rat.add_comm failureMass successMass, partition,
              Rat.one_mul]
      have powerNonnegative :
          0 <= failureMass ^ smaller :=
        Rat.pow_nonneg failureNonnegative
      change
        failureMass ^ (smaller + 1) *
            (1 + ((smaller + 1 : Nat) : Rat) * successMass) <= 1
      rw [Rat.pow_succ]
      have linearSucc :
          1 + ((smaller + 1 : Nat) : Rat) * successMass =
            linear + successMass := by
        unfold linear
        simp [Rat.natCast_add, Rat.add_mul, Rat.add_assoc]
      rw [linearSucc]
      calc
        failureMass ^ smaller * failureMass *
              (linear + successMass) =
            failureMass ^ smaller *
              (failureMass * (linear + successMass)) := by
          rw [Rat.mul_assoc]
        _ <= failureMass ^ smaller * linear :=
          Rat.mul_le_mul_of_nonneg_left branchBound powerNonnegative
        _ <= 1 := inductionHypothesis

/-- The exact geometric tail is bounded by the reciprocal of its linear
success drift. This elementary inequality is sufficient to prove tail
vanishing without importing a general measure or convergence library. -/
theorem failureTail_le_inverse_linear
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptCount : Nat) :
    failureTail experiment success attemptCount <=
      1 /
        (1 + (attemptCount : Rat) *
          successProbability experiment success) := by
  let successMass := successProbability experiment success
  let failureMass := failureProbability experiment success
  have successNonnegative : 0 <= successMass :=
    experiment.probabilityBool_nonneg success
  have failureNonnegative : 0 <= failureMass :=
    experiment.probabilityBool_nonneg fun outcome => !success outcome
  have partition : successMass + failureMass = 1 :=
    success_add_failure experiment success
  have denominatorPositive :
      0 < 1 + (attemptCount : Rat) * successMass := by
    apply rat_lt_of_lt_of_le (by decide : (0 : Rat) < 1)
    simpa only [Rat.add_zero] using
      ((Rat.add_le_add_left (c := 1)).mpr
        (Rat.mul_nonneg Rat.natCast_nonneg successNonnegative))
  unfold failureTail
  exact
    (le_div_iff_of_pos denominatorPositive).mpr
      (geometricTail_mul_linear_le_one
        successMass failureMass successNonnegative failureNonnegative
        partition attemptCount)

private theorem inverse_cardinality_le_successProbability
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    1 / (experiment.support.cardinality : Rat) <=
      successProbability experiment success := by
  have countPositive : 0 < experiment.countBool success := by
    unfold Experiment.countBool
    rw [List.countP_eq_length_filter]
    exact List.length_pos_iff.mpr nonempty
  have oneLeCount : 1 <= experiment.countBool success :=
    Nat.succ_le_iff.mpr countPositive
  unfold successProbability Experiment.probabilityBool
  exact div_le_div_of_le
    (Rat.natCast_le_natCast.mpr oneLeCount)
    (Rat.natCast_pos.mpr experiment.support.cardinality_pos)

/-- The failure tail becomes smaller than every inverse-natural threshold.
The explicit witness uses `threshold * |support|` attempts. -/
theorem failureTail_le_inverseNatural
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (threshold : Nat) :
    failureTail experiment success
        (threshold * experiment.support.cardinality) <=
      1 / ((threshold + 1 : Nat) : Rat) := by
  let cardinality := experiment.support.cardinality
  let successMass := successProbability experiment success
  have cardinalityPositive : 0 < (cardinality : Rat) :=
    Rat.natCast_pos.mpr experiment.support.cardinality_pos
  have inverseLower :
      1 / (cardinality : Rat) <= successMass :=
    inverse_cardinality_le_successProbability
      experiment success nonempty
  have cardinalityTimesSuccess :
      1 <= (cardinality : Rat) * successMass := by
    calc
      1 = (cardinality : Rat) *
          (1 / (cardinality : Rat)) := by
        rw [Rat.mul_comm, Rat.div_mul_cancel
          (Rat.ne_of_gt cardinalityPositive)]
      _ <= (cardinality : Rat) * successMass :=
        Rat.mul_le_mul_of_nonneg_left inverseLower Rat.natCast_nonneg
  have thresholdTimesSuccess :
      (threshold : Rat) <=
        ((threshold * cardinality : Nat) : Rat) * successMass := by
    calc
      (threshold : Rat) = (threshold : Rat) * 1 := (Rat.mul_one _).symm
      _ <= (threshold : Rat) *
          ((cardinality : Rat) * successMass) :=
        Rat.mul_le_mul_of_nonneg_left
          cardinalityTimesSuccess Rat.natCast_nonneg
      _ =
          ((threshold * cardinality : Nat) : Rat) * successMass := by
        rw [Rat.natCast_mul, Rat.mul_assoc]
  have denominatorBound :
      ((threshold + 1 : Nat) : Rat) <=
        1 + ((threshold * cardinality : Nat) : Rat) * successMass := by
    rw [Rat.natCast_add]
    change
      (threshold : Rat) + 1 <=
        1 + ((threshold * cardinality : Nat) : Rat) * successMass
    rw [Rat.add_comm (threshold : Rat) 1]
    exact (Rat.add_le_add_left (c := 1)).mpr thresholdTimesSuccess
  exact Rat.le_trans
    (failureTail_le_inverse_linear experiment success
      (threshold * cardinality))
    (div_le_div_of_nonneg_of_le_of_pos_le
      (by decide : (0 : Rat) <= 1) Rat.le_refl
      (Rat.natCast_pos.mpr (Nat.succ_pos threshold))
      denominatorBound)

private theorem inverse_den_succ_lt
    {epsilon : Rat}
    (epsilonPositive : 0 < epsilon) :
    1 / ((epsilon.den + 1 : Nat) : Rat) < epsilon := by
  have denominatorPositive : 0 < (epsilon.den : Rat) :=
    Rat.natCast_pos.mpr (Nat.pos_of_ne_zero epsilon.den_nz)
  have successorDenominatorPositive :
      0 < ((epsilon.den + 1 : Nat) : Rat) :=
    Rat.natCast_pos.mpr (Nat.succ_pos epsilon.den)
  have inverseStrict :
      1 / ((epsilon.den + 1 : Nat) : Rat) <
        1 / (epsilon.den : Rat) := by
    apply (Rat.div_lt_iff successorDenominatorPositive).mpr
    calc
      (1 : Rat) =
          (1 / (epsilon.den : Rat)) * (epsilon.den : Rat) := by
        rw [Rat.div_mul_cancel (Rat.ne_of_gt denominatorPositive)]
      _ <
          (1 / (epsilon.den : Rat)) *
            ((epsilon.den + 1 : Nat) : Rat) :=
        Rat.mul_lt_mul_of_pos_left
          (Rat.natCast_lt_natCast.mpr (Nat.lt_succ_self epsilon.den))
          (by
            rw [Rat.div_def, Rat.one_mul]
            exact Rat.inv_pos.mpr denominatorPositive)
  have numeratorPositive : (0 : Int) < epsilon.num := by
    simpa [Rat.lt_iff] using epsilonPositive
  have oneLeNumeratorInt : (1 : Int) <= epsilon.num :=
    Int.add_one_le_iff.mpr numeratorPositive
  have oneLeNumerator : (1 : Rat) <= (epsilon.num : Rat) :=
    Rat.intCast_le_intCast.mpr oneLeNumeratorInt
  have inverseLeEpsilon :
      1 / (epsilon.den : Rat) <= epsilon := by
    calc
      1 / (epsilon.den : Rat) <=
          (epsilon.num : Rat) / (epsilon.den : Rat) :=
        div_le_div_of_le oneLeNumerator denominatorPositive
      _ = epsilon := by
        exact (Rat.divInt_eq_div epsilon.num (epsilon.den : Int)).symm.trans
          (Rat.num_divInt_den epsilon)
  exact rat_lt_of_lt_of_le inverseStrict inverseLeEpsilon

/-- Vanishing of the residual all-failure probability in the exact rational
epsilon sense. -/
def FailureTailVanishes
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Prop :=
  forall epsilon : Rat, 0 < epsilon ->
    exists attemptCount,
      failureTail experiment success attemptCount < epsilon

/-- Positive one-run success derives an explicit geometric tail limit. -/
theorem failureTail_vanishes
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    FailureTailVanishes experiment success := by
  intro epsilon epsilonPositive
  refine
    ⟨epsilon.den * experiment.support.cardinality,
      rat_lt_of_le_of_lt
        (failureTail_le_inverseNatural
          experiment success nonempty epsilon.den)
        (inverse_den_succ_lt epsilonPositive)⟩

/-- Closed form of the countable terminating trace mass. -/
def totalTerminationMass
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Rat :=
  successProbability experiment success /
    (1 - failureProbability experiment success)

/-- The specialized unbounded trace law has total mass one whenever the
successful support is nonempty. -/
theorem trace_totalMass_eq_one
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    totalTerminationMass experiment success = 1 := by
  have successPos :=
    successProbability_pos experiment success nonempty
  have denominator :
      1 - failureProbability experiment success =
        successProbability experiment success := by
    rw [failure_eq_one_sub_success]
    simp only [Rat.sub_eq_add_neg, Rat.neg_add, Rat.neg_neg]
    rw [← Rat.add_assoc, Rat.add_neg_cancel, Rat.zero_add]
  unfold totalTerminationMass
  rw [denominator, Rat.div_def,
    Rat.mul_inv_cancel _ (Rat.ne_of_gt successPos)]

/-- Almost-sure termination for this discrete law records both normalization
of finite terminating traces and vanishing residual all-failure mass. -/
def AlmostSurelyTerminates
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Prop :=
  totalTerminationMass experiment success = 1 /\
    FailureTailVanishes experiment success

/-- Positive one-run success derives almost-sure termination. -/
theorem firstSuccess_terminates_almostSurely
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    AlmostSurelyTerminates experiment success :=
  ⟨trace_totalMass_eq_one experiment success nonempty,
    failureTail_vanishes experiment success nonempty⟩

/-- Event seen when the current retry succeeds and the independent fresh run
has also been drawn. -/
def successfulBranchEvent
    (success : Outcome -> Bool)
    (event : Outcome × Outcome -> Bool) :
    Outcome × Outcome -> Bool :=
  fun sample => success sample.1 && event sample

/-- Probability contributed by the successful branch of one retry step. -/
def successfulBranchProbability
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (event : Outcome × Outcome -> Bool) : Rat :=
  experiment.iidPair.probabilityBool
    (successfulBranchEvent success event)

/-- Unbounded first-success/fresh-second event law, obtained by solving its
geometric first-step recursion. -/
def jointProbability
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (event : Outcome × Outcome -> Bool) : Rat :=
  successfulBranchProbability experiment success event /
    successProbability experiment success

private theorem conditioned_event_insert_success
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        event =
      (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        (successfulBranchEvent success event) := by
  let filtered :=
    experiment.support.values.filter
      (fun seed => success (experiment.outcome seed))
  change
    (filtered.map (fun firstSeed =>
      experiment.probabilityBool (fun second =>
        event (experiment.outcome firstSeed, second)))).sum /
        (filtered.length : Rat) =
      (filtered.map (fun firstSeed =>
        experiment.probabilityBool (fun second =>
          successfulBranchEvent success event
            (experiment.outcome firstSeed, second)))).sum /
        (filtered.length : Rat)
  apply congrArg (fun numerator : Rat =>
    numerator / (filtered.length : Rat))
  apply congrArg List.sum
  apply List.map_congr_left
  intro firstSeed firstMember
  have firstAccepted :
      success (experiment.outcome firstSeed) = true :=
    (List.mem_filter.mp firstMember).2
  congr 1
  funext second
  simp [successfulBranchEvent, firstAccepted]

/-- The unbounded trace output law is exactly the success-conditioned first
execution paired with one fresh, unconditioned execution. -/
theorem jointProbability_eq_firstConditionedFreshSecond
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool) :
    jointProbability experiment success event =
      (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        event := by
  have divided :=
    Experiment.firstConditionedFreshSecond_probabilityBool_eq_div
      experiment success nonempty
      (successfulBranchEvent success event)
      (by
        intro first second eventHolds
        unfold successfulBranchEvent at eventHolds
        exact (Bool.and_eq_true_iff.mp eventHolds).1)
  calc
    jointProbability experiment success event =
        (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
          (successfulBranchEvent success event) := divided.symm
    _ =
        (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
          event :=
      (conditioned_event_insert_success
        experiment success nonempty event).symm

/-- Explicit finite product representation of the selected-first and fresh
second output law. Its seed support is literally successful first seeds
times the complete original support. -/
def conditionedFreshProduct
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    Experiment (Outcome × Outcome) where
  Seed := experiment.Seed × experiment.Seed
  support :=
    (experiment.support.filterBool
      (fun seed => success (experiment.outcome seed)) nonempty).product
        experiment.support
  outcome := fun seeds =>
    (experiment.outcome seeds.1, experiment.outcome seeds.2)

/-- Exact fresh-second independence: the unbounded sampler law is the
Cartesian product of the success-filtered first support and the unfiltered
one-run support. -/
theorem jointProbability_eq_conditionedFreshProduct
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool) :
    jointProbability experiment success event =
      (conditionedFreshProduct experiment success nonempty).probabilityBool
        event := by
  rw [jointProbability_eq_firstConditionedFreshSecond
    experiment success nonempty event]
  exact Mixture.sharedSupport_probabilityBool_eq_product
    (experiment.support.filterBool
      (fun seed => success (experiment.outcome seed)) nonempty)
    experiment.support
    (fun firstSeed secondSeed =>
      (experiment.outcome firstSeed, experiment.outcome secondSeed))
    event

/-- The fresh second marginal is exactly the original unconditioned one-run
law. -/
theorem freshSecond_marginal
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome -> Bool) :
    jointProbability experiment success (fun sample => event sample.2) =
      experiment.probabilityBool event := by
  rw [jointProbability_eq_firstConditionedFreshSecond
    experiment success nonempty]
  exact experiment.firstConditionedFreshSecond_second_marginal
    success nonempty event

/-- The exact expected number of retry executions before and including the
first success. -/
def expectedRetryExecutions
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Rat :=
  1 / successProbability experiment success

/-- The exact expected number of complete executions after also charging the
fresh second run. -/
def expectedTotalExecutions
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) : Rat :=
  expectedRetryExecutions experiment success + 1

private theorem geometric_fixedPoint
    (successMass branchMass : Rat)
    (successMassNe : successMass ≠ 0) :
    branchMass / successMass =
      branchMass +
        (1 - successMass) * (branchMass / successMass) := by
  have cancel :
      successMass * (branchMass / successMass) = branchMass := by
    rw [Rat.mul_comm, Rat.div_mul_cancel successMassNe]
  calc
    branchMass / successMass =
        branchMass + branchMass / successMass - branchMass := by
      rw [Rat.add_comm branchMass, Rat.add_sub_cancel]
    _ =
        branchMass + branchMass / successMass -
          successMass * (branchMass / successMass) := by
      rw [cancel]
    _ =
        branchMass +
          (1 - successMass) * (branchMass / successMass) := by
      simp only [Rat.sub_eq_add_neg, Rat.add_mul, Rat.one_mul, Rat.neg_mul]
      rw [Rat.add_assoc]

/-- Positive success makes the restart equation's finite solution unique. -/
private theorem geometric_fixedPoint_unique
    (successMass branchMass candidate : Rat)
    (successMassNe : successMass ≠ 0)
    (fixedPoint :
      candidate =
        branchMass + (1 - successMass) * candidate) :
    candidate = branchMass / successMass := by
  have rearranged :
      candidate + successMass * candidate =
        branchMass + candidate := by
    calc
      candidate + successMass * candidate =
          (branchMass + (1 - successMass) * candidate) +
            successMass * candidate :=
        congrArg (fun value => value + successMass * candidate) fixedPoint
      _ =
          (branchMass +
              (candidate + -(successMass * candidate))) +
            successMass * candidate := by
        simp only [Rat.sub_eq_add_neg, Rat.add_mul, Rat.one_mul, Rat.neg_mul]
      _ = branchMass + candidate := by
        rw [Rat.add_assoc branchMass,
          Rat.add_assoc candidate,
          Rat.neg_add_cancel, Rat.add_zero]
  have scaled : successMass * candidate = branchMass := by
    apply Rat.add_left_cancel candidate
    calc
      candidate + successMass * candidate =
          branchMass + candidate := rearranged
      _ = candidate + branchMass := Rat.add_comm _ _
  calc
    candidate = candidate * successMass / successMass :=
      (Rat.mul_div_cancel successMassNe).symm
    _ = successMass * candidate / successMass := by
      rw [Rat.mul_comm candidate successMass]
    _ = branchMass / successMass := by
      rw [scaled]

/-- Event probabilities satisfy the operational first-step retry equation:
take the successful branch now, or fail and restart the same sampler. -/
theorem jointProbability_firstStep
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool) :
    jointProbability experiment success event =
      successfulBranchProbability experiment success event +
        failureProbability experiment success *
          jointProbability experiment success event := by
  have successNe :
      successProbability experiment success ≠ 0 :=
    Rat.ne_of_gt (successProbability_pos experiment success nonempty)
  unfold jointProbability
  rw [failure_eq_one_sub_success]
  exact geometric_fixedPoint
    (successProbability experiment success)
    (successfulBranchProbability experiment success event)
    successNe

/-- No other finite event weight can satisfy the same first-step sampler
equation. -/
theorem jointProbability_unique
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool)
    (candidate : Rat)
    (candidateFirstStep :
      candidate =
        successfulBranchProbability experiment success event +
          failureProbability experiment success * candidate) :
    candidate = jointProbability experiment success event := by
  have successNe :
      successProbability experiment success ≠ 0 :=
    Rat.ne_of_gt (successProbability_pos experiment success nonempty)
  unfold jointProbability
  apply geometric_fixedPoint_unique
    (successProbability experiment success)
    (successfulBranchProbability experiment success event)
    candidate successNe
  rw [← failure_eq_one_sub_success experiment success]
  exact candidateFirstStep

/-- The geometric expected retry count satisfies the same restart equation:
one current execution, followed by another copy only on failure. -/
theorem expectedRetryExecutions_firstStep
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    expectedRetryExecutions experiment success =
      1 + failureProbability experiment success *
        expectedRetryExecutions experiment success := by
  have successNe :
      successProbability experiment success ≠ 0 :=
    Rat.ne_of_gt (successProbability_pos experiment success nonempty)
  unfold expectedRetryExecutions
  rw [failure_eq_one_sub_success]
  simpa using
    geometric_fixedPoint
      (successProbability experiment success) 1 successNe

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace
