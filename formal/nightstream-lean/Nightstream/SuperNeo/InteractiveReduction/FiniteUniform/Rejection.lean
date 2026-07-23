import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Finite rejection conditioning over the existing uniform-experiment model.

Owns: executable filtering of a nonempty finite support, conditioning the first
run on a Boolean success event, an independent fresh second run, the exact
finite conditional-pair ratio, denominator-floor monotonicity, and finite
Boolean union/subtractive accounting.

Does not own: a protocol, coordinate forking, an infinite or geometric
rejection sampler, asymptotic running time, Fiat--Shamir, Rust, R1CS, or
constraints.

The first-conditioned experiment is an explicit uniform mixture.  Its outer
support contains exactly the successful first seeds; every component executes
the original experiment afresh for the second run.  No product support and no
second probability model are introduced.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uSeed uOutcome

variable {Seed : Type uSeed}
variable {Outcome : Type uOutcome}

/-- Filter a finite support by an executable predicate.  The caller exposes
the semantically necessary fact that the conditioned support is nonempty. -/
def Support.filterBool
    (support : Support Seed)
    (keep : Seed -> Bool)
    (nonempty : support.values.filter keep ≠ []) : Support Seed where
  values := support.values.filter keep
  nodup := List.Nodup.sublist List.filter_sublist support.nodup
  nonempty := nonempty

@[simp] theorem Support.filterBool_values
    (support : Support Seed)
    (keep : Seed -> Bool)
    (nonempty : support.values.filter keep ≠ []) :
    (support.filterBool keep nonempty).values = support.values.filter keep :=
  rfl

theorem Support.filterBool_cardinality
    (support : Support Seed)
    (keep : Seed -> Bool)
    (nonempty : support.values.filter keep ≠ []) :
    (support.filterBool keep nonempty).cardinality =
      support.values.countP keep := by
  unfold Support.cardinality Support.filterBool
  exact List.countP_eq_length_filter.symm

/-- Restrict an experiment to the seeds whose outputs satisfy `success`. -/
def Experiment.conditionBool
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    Experiment Outcome where
  Seed := experiment.Seed
  support := experiment.support.filterBool
    (fun seed => success (experiment.outcome seed)) nonempty
  outcome := experiment.outcome

theorem Experiment.conditionBool_success
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    forall seed,
      seed ∈ (experiment.conditionBool success nonempty).support.values ->
        success ((experiment.conditionBool success nonempty).outcome seed) =
          true := by
  intro seed member
  exact (List.mem_filter.mp member).2

private theorem rat_cardinality_pos
    (support : Support Seed) :
    0 < (support.cardinality : Rat) := by
  exact Rat.natCast_pos.mpr support.cardinality_pos

private theorem rat_cardinality_ne_zero
    (support : Support Seed) :
    (support.cardinality : Rat) ≠ 0 :=
  Rat.ne_of_gt (rat_cardinality_pos support)

theorem Experiment.probabilityBool_nonneg
    (experiment : Experiment Outcome)
    (event : Outcome -> Bool) :
    0 <= experiment.probabilityBool event := by
  unfold Experiment.probabilityBool
  rw [Rat.div_def]
  exact Rat.mul_nonneg Rat.natCast_nonneg
    (Rat.le_of_lt (Rat.inv_pos.mpr
      (rat_cardinality_pos experiment.support)))

theorem Experiment.probabilityBool_pos_of_filter_nonempty
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    0 < experiment.probabilityBool success := by
  have countPos : 0 < experiment.countBool success := by
    unfold Experiment.countBool
    rw [List.countP_eq_length_filter]
    exact List.length_pos_iff.mpr nonempty
  unfold Experiment.probabilityBool
  rw [Rat.div_def]
  exact Rat.mul_pos (Rat.natCast_pos.mpr countPos)
    (Rat.inv_pos.mpr (rat_cardinality_pos experiment.support))

/-- Two independent executions represented as a uniform mixture over the
first seed and a fresh copy of the original experiment for the second. -/
def Experiment.iidPair
    (experiment : Experiment Outcome) :
    Mixture experiment.Seed (Outcome × Outcome) where
  prefixes := experiment.support
  component := fun firstSeed =>
    experiment.map fun second => (experiment.outcome firstSeed, second)

/-- Appendix D.4's finite experiment: the first seed is sampled uniformly
from successful seeds and the second execution is fresh and unconditioned. -/
def Experiment.firstConditionedFreshSecond
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    Mixture experiment.Seed (Outcome × Outcome) where
  prefixes := experiment.support.filterBool
    (fun seed => success (experiment.outcome seed)) nonempty
  component := fun firstSeed =>
    experiment.map fun second => (experiment.outcome firstSeed, second)

theorem Mixture.probabilityBool_nonneg
    {Prefix : Type uSeed}
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Bool) :
    0 <= mixture.probabilityBool event := by
  have sumNonneg : 0 <=
      (mixture.prefixes.values.map
        (fun outer => (mixture.component outer).probabilityBool event)).sum := by
    induction mixture.prefixes.values with
    | nil => exact Rat.le_refl
    | cons head tail inductionHypothesis =>
        simp only [List.map_cons, List.sum_cons]
        exact Rat.add_nonneg
          ((mixture.component head).probabilityBool_nonneg event)
          inductionHypothesis
  unfold Mixture.probabilityBool
  rw [Rat.div_def]
  exact Rat.mul_nonneg sumNonneg
    (Rat.le_of_lt (Rat.inv_pos.mpr
      (rat_cardinality_pos mixture.prefixes)))

/-- One-line adapter from the executable iid-pair event to the existing
mathematical-event probability. -/
theorem Experiment.iidPair_probability_bool_event
    (experiment : Experiment Outcome)
    (event : Outcome × Outcome -> Bool) :
    experiment.iidPair.probability
        (fun sample => event sample = true) =
      experiment.iidPair.probabilityBool event :=
  experiment.iidPair.probability_bool_event event

/-- One-line adapter from the executable conditioned-pair event to the
existing mathematical-event probability. -/
theorem Experiment.firstConditionedFreshSecond_probability_bool_event
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool) :
    (experiment.firstConditionedFreshSecond success nonempty).probability
        (fun sample => event sample = true) =
      (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        event :=
  Mixture.probability_bool_event
    (experiment.firstConditionedFreshSecond success nonempty) event

private theorem Experiment.probabilityBool_eq_zero_of_seedwise_false
    (experiment : Experiment Outcome)
    (event : Outcome -> Bool)
    (allFalse : forall seed, seed ∈ experiment.support.values ->
      event (experiment.outcome seed) = false) :
    experiment.probabilityBool event = 0 := by
  have countZero : experiment.countBool event = 0 := by
    unfold Experiment.countBool
    apply List.countP_eq_zero.mpr
    intro seed member
    rw [allFalse seed member]
    intro impossible
    exact Bool.noConfusion impossible
  unfold Experiment.probabilityBool
  rw [countZero]
  simp [Rat.div_def]

private theorem sum_map_filter_eq_sum_map_of_false_zero
    {Element : Type uSeed}
    (values : List Element)
    (keep : Element -> Bool)
    (value : Element -> Rat)
    (zeroOutside : forall element, element ∈ values ->
      keep element = false -> value element = 0) :
    ((values.filter keep).map value).sum = (values.map value).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases keepHead : keep head with
      | false =>
          have headZero : value head = 0 :=
            zeroOutside head (by simp) keepHead
          simp only [List.filter_cons, keepHead, Bool.false_eq_true,
            ↓reduceIte, List.map_cons, List.sum_cons, headZero, Rat.zero_add]
          exact inductionHypothesis (by
            intro element member keptFalse
            exact zeroOutside element (by simp [member]) keptFalse)
      | true =>
          simp only [List.filter_cons, keepHead, ↓reduceIte, List.map_cons,
            List.sum_cons]
          rw [inductionHypothesis]
          intro element member keptFalse
          exact zeroOutside element (by simp [member]) keptFalse

private theorem div_div_same_denominator
    (numerator firstDenominator commonDenominator : Rat)
    (commonDenominatorNe : commonDenominator ≠ 0) :
    (numerator / commonDenominator) /
        (firstDenominator / commonDenominator) =
      numerator / firstDenominator := by
  rw [Rat.div_def, Rat.div_def, Rat.div_def, Rat.inv_mul_rev,
    Rat.inv_inv]
  calc
    numerator * commonDenominator⁻¹ *
          (commonDenominator * firstDenominator⁻¹) =
        numerator * (commonDenominator⁻¹ * commonDenominator) *
          firstDenominator⁻¹ := by
            rw [← Rat.mul_assoc
              (numerator * commonDenominator⁻¹)
              commonDenominator firstDenominator⁻¹,
              Rat.mul_assoc numerator commonDenominator⁻¹ commonDenominator]
    _ = numerator * firstDenominator⁻¹ := by
      rw [Rat.inv_mul_cancel commonDenominator commonDenominatorNe,
        Rat.mul_one]

/-- Exact finite conditioning identity.  The event premise says that an event
counted in the raw iid experiment can occur only when the first run satisfies
the conditioning predicate. -/
theorem Experiment.firstConditionedFreshSecond_probabilityBool_eq_div
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool)
    (eventImpliesFirstSuccess : forall first second,
      event (first, second) = true -> success first = true) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        event =
      experiment.iidPair.probabilityBool event /
        experiment.probabilityBool success := by
  let keep : experiment.Seed -> Bool :=
    fun seed => success (experiment.outcome seed)
  let componentProbability : experiment.Seed -> Rat :=
    fun firstSeed => experiment.probabilityBool fun second =>
      event (experiment.outcome firstSeed, second)
  have zeroOutside : forall firstSeed,
      firstSeed ∈ experiment.support.values ->
      keep firstSeed = false -> componentProbability firstSeed = 0 := by
    intro firstSeed _ firstRejected
    apply Experiment.probabilityBool_eq_zero_of_seedwise_false
    intro secondSeed _secondMember
    cases eventHolds :
        event (experiment.outcome firstSeed, experiment.outcome secondSeed) with
    | false => rfl
    | true =>
        have firstAccepted := eventImpliesFirstSuccess
          (experiment.outcome firstSeed) (experiment.outcome secondSeed)
          eventHolds
        change success (experiment.outcome firstSeed) = false at firstRejected
        rw [firstAccepted] at firstRejected
        exact Bool.noConfusion firstRejected
  have filteredSum :
      (((experiment.support.values.filter keep).map
        componentProbability).sum) =
        (experiment.support.values.map componentProbability).sum :=
    sum_map_filter_eq_sum_map_of_false_zero
      experiment.support.values keep componentProbability zeroOutside
  have originalCardinalityNe :
      (experiment.support.values.length : Rat) ≠ 0 := by
    exact rat_cardinality_ne_zero experiment.support
  change
    (((experiment.support.values.filter keep).map componentProbability).sum /
        ((experiment.support.values.filter keep).length : Rat)) =
      ((experiment.support.values.map componentProbability).sum /
          (experiment.support.values.length : Rat)) /
        ((experiment.support.values.countP keep : Rat) /
          (experiment.support.values.length : Rat))
  rw [filteredSum, List.countP_eq_length_filter]
  exact (div_div_same_denominator
    (experiment.support.values.map componentProbability).sum
    ((experiment.support.values.filter keep).length : Rat)
    (experiment.support.values.length : Rat)
    originalCardinalityNe).symm

/-- Increasing the numerator budget and decreasing a positive denominator
can only increase the resulting ratio.  Positivity of the actual denominator
is derived explicitly from the positive floor and its lower-bound proof. -/
theorem div_le_div_of_nonneg_of_le_of_pos_le
    {raw budget actual floor : Rat}
    (rawNonneg : 0 <= raw)
    (rawBound : raw <= budget)
    (floorPos : 0 < floor)
    (floorBound : floor <= actual) :
    raw / actual <= budget / floor := by
  have actualPos : 0 < actual := by
    apply Rat.not_le.mp
    intro actualNonpos
    exact (Rat.not_le.mpr floorPos) (Rat.le_trans floorBound actualNonpos)
  have budgetNonneg : 0 <= budget := Rat.le_trans rawNonneg rawBound
  have floorInverseNonneg : 0 <= floor⁻¹ :=
    Rat.le_of_lt (Rat.inv_pos.mpr floorPos)
  have budgetOverFloorNonneg : 0 <= budget / floor := by
    rw [Rat.div_def]
    exact Rat.mul_nonneg budgetNonneg floorInverseNonneg
  have sameDenominator : raw / actual <= budget / actual :=
    div_le_div_of_le rawBound actualPos
  have denominatorOrder : budget / actual <= budget / floor := by
    apply (div_le_iff_of_pos actualPos).mpr
    have scaled := Rat.mul_le_mul_of_nonneg_left floorBound
      budgetOverFloorNonneg
    rw [Rat.div_mul_cancel (Rat.ne_of_gt floorPos)] at scaled
    exact scaled
  exact Rat.le_trans sameDenominator denominatorOrder

/-- Appendix D.4's raw disagreement bound divided by any positive lower bound
on first-run success. -/
theorem Experiment.firstConditionedFreshSecond_probabilityBool_le_div_floor
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool)
    (eventImpliesFirstSuccess : forall first second,
      event (first, second) = true -> success first = true)
    (budget floor : Rat)
    (rawBound : experiment.iidPair.probabilityBool event <= budget)
    (floorPos : 0 < floor)
    (floorBound : floor <= experiment.probabilityBool success) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        event <= budget / floor := by
  rw [experiment.firstConditionedFreshSecond_probabilityBool_eq_div success
    nonempty event eventImpliesFirstSuccess]
  exact div_le_div_of_nonneg_of_le_of_pos_le
    (Mixture.probabilityBool_nonneg experiment.iidPair event)
    rawBound floorPos floorBound

private theorem sum_map_le_sum_map_local
    {Element : Type uSeed}
    (values : List Element)
    (left right : Element -> Rat)
    (ordered : forall element, element ∈ values ->
      left element <= right element) :
    (values.map left).sum <= (values.map right).sum := by
  induction values with
  | nil => exact Rat.le_refl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons]
      exact Rat.le_trans
        ((Rat.add_le_add_right (c := (tail.map left).sum)).mpr
          (ordered head (by simp)))
        ((Rat.add_le_add_left (c := right head)).mpr
          (inductionHypothesis (by
            intro element member
            exact ordered element (by simp [member]))))

private theorem sum_map_constant
    {Element : Type uSeed}
    (values : List Element)
    (constant : Rat) :
    (values.map (fun _ => constant)).sum =
      (values.length : Rat) * constant := by
  induction values with
  | nil => simp
  | cons _ tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        Rat.natCast_add, inductionHypothesis,
        Rat.natCast_ofNat, Rat.add_mul, Rat.one_mul]
      rw [Rat.add_comm]

/-- A pointwise bound for every fixed successful first seed survives the
uniform outer mixture without multiplication by the number of first seeds. -/
theorem Experiment.firstConditionedFreshSecond_probabilityBool_le_of_fixedFirst
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (event : Outcome × Outcome -> Bool)
    (bound : Rat)
    (componentBound : forall firstSeed,
      firstSeed ∈ experiment.support.values.filter
        (fun seed => success (experiment.outcome seed)) ->
      experiment.probabilityBool (fun second =>
        event (experiment.outcome firstSeed, second)) <= bound) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
      event <= bound := by
  let filteredSeeds := experiment.support.values.filter
    (fun seed => success (experiment.outcome seed))
  let componentProbability : experiment.Seed -> Rat :=
    fun firstSeed => experiment.probabilityBool fun second =>
      event (experiment.outcome firstSeed, second)
  have sumBound :
      (filteredSeeds.map componentProbability).sum <=
        (filteredSeeds.map (fun _ => bound)).sum := by
    apply sum_map_le_sum_map_local
    intro firstSeed member
    exact componentBound firstSeed member
  have lengthPos : 0 < filteredSeeds.length := by
    exact List.length_pos_iff.mpr nonempty
  have lengthRatPos : 0 < (filteredSeeds.length : Rat) :=
    Rat.natCast_pos.mpr lengthPos
  change (filteredSeeds.map componentProbability).sum /
      (filteredSeeds.length : Rat) <= bound
  have divided := div_le_div_of_le sumBound lengthRatPos
  rw [sum_map_constant,
    Rat.mul_comm (filteredSeeds.length : Rat) bound,
    Rat.mul_div_cancel (Rat.ne_of_gt lengthRatPos)] at divided
  exact divided

/-- Conditioning the first run does not change the marginal distribution of
the fresh second run. -/
theorem Experiment.firstConditionedFreshSecond_second_marginal
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (secondEvent : Outcome -> Bool) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        (fun sample => secondEvent sample.2) =
      experiment.probabilityBool secondEvent := by
  let filteredSeeds := experiment.support.values.filter
    (fun seed => success (experiment.outcome seed))
  have lengthPos : 0 < filteredSeeds.length :=
    List.length_pos_iff.mpr nonempty
  change
    (filteredSeeds.map
      (fun _ => experiment.probabilityBool secondEvent)).sum /
        (filteredSeeds.length : Rat) =
      experiment.probabilityBool secondEvent
  rw [sum_map_constant,
    Rat.mul_comm (filteredSeeds.length : Rat)
      (experiment.probabilityBool secondEvent)]
  exact Rat.mul_div_cancel
    (Rat.ne_of_gt (Rat.natCast_pos.mpr lengthPos))

private theorem Experiment.probabilityBool_true
    (experiment : Experiment Outcome) :
    experiment.probabilityBool (fun _ => true) = 1 := by
  rw [← experiment.probability_bool_event]
  simpa using experiment.probability_true

private theorem sum_map_bool_indicator_eq_countP
    {Element : Type uSeed}
    (values : List Element)
    (event : Element -> Bool) :
    (values.map (fun element =>
      if event element then (1 : Rat) else 0)).sum =
        (values.countP event : Rat) := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases headEvent : event head with
      | false =>
          simp [headEvent, inductionHypothesis]
          exact Rat.zero_add _
      | true =>
          simp only [List.map_cons, List.sum_cons, List.countP_cons,
            headEvent, ↓reduceIte, Rat.natCast_add, Rat.natCast_ofNat,
            inductionHypothesis]
          rw [Rat.add_comm]

/-- The first marginal of the unconditioned iid pair is exactly the original
experiment.  The proof counts first seeds; it does not enumerate a product
support. -/
theorem Experiment.iidPair_first_marginal
    (experiment : Experiment Outcome)
    (firstEvent : Outcome -> Bool) :
    experiment.iidPair.probabilityBool
        (fun sample => firstEvent sample.1) =
      experiment.probabilityBool firstEvent := by
  let componentProbability : experiment.Seed -> Rat :=
    fun firstSeed => experiment.probabilityBool fun _ =>
      firstEvent (experiment.outcome firstSeed)
  have componentMap :
      experiment.support.values.map componentProbability =
        experiment.support.values.map (fun firstSeed =>
          if firstEvent (experiment.outcome firstSeed) then (1 : Rat) else 0) := by
    apply List.map_congr_left
    intro firstSeed _
    cases firstHolds : firstEvent (experiment.outcome firstSeed) with
    | false =>
        unfold componentProbability
        rw [firstHolds]
        apply Experiment.probabilityBool_eq_zero_of_seedwise_false
        intro _ _
        rfl
    | true =>
        unfold componentProbability
        rw [firstHolds]
        exact experiment.probabilityBool_true
  change (experiment.support.values.map componentProbability).sum /
      (experiment.support.cardinality : Rat) =
    (experiment.support.values.countP
      (fun seed => firstEvent (experiment.outcome seed)) : Rat) /
        (experiment.support.cardinality : Rat)
  rw [componentMap, sum_map_bool_indicator_eq_countP]

/-- The conditioning event holds with probability one in the conditioned
first-run marginal.  This is derived from membership in the filtered support,
not accepted as a premise. -/
theorem Experiment.firstConditionedFreshSecond_first_success
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        (fun sample => success sample.1) = 1 := by
  let filteredSeeds := experiment.support.values.filter
    (fun seed => success (experiment.outcome seed))
  let componentProbability : experiment.Seed -> Rat :=
    fun firstSeed => experiment.probabilityBool fun _ =>
      success (experiment.outcome firstSeed)
  have componentMap :
      filteredSeeds.map componentProbability =
        filteredSeeds.map (fun _ => (1 : Rat)) := by
    apply List.map_congr_left
    intro firstSeed member
    have firstAccepted : success (experiment.outcome firstSeed) = true :=
      (List.mem_filter.mp member).2
    unfold componentProbability
    rw [firstAccepted]
    exact experiment.probabilityBool_true
  have lengthPos : 0 < filteredSeeds.length :=
    List.length_pos_iff.mpr nonempty
  change (filteredSeeds.map componentProbability).sum /
      (filteredSeeds.length : Rat) = 1
  rw [componentMap, sum_map_constant, Rat.mul_one, Rat.div_def,
    Rat.mul_inv_cancel _ (Rat.ne_of_gt (Rat.natCast_pos.mpr lengthPos))]

private theorem countP_or_le
    {Element : Type uSeed}
    (values : List Element)
    (left right : Element -> Bool) :
    values.countP (fun element => left element || right element) <=
      values.countP left + values.countP right := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases leftHead : left head <;> cases rightHead : right head
      · simpa [List.countP_cons, leftHead, rightHead] using
          inductionHypothesis
      · simpa [List.countP_cons, leftHead, rightHead, Nat.add_assoc,
          Nat.add_comm, Nat.add_left_comm] using
          Nat.add_le_add_right inductionHypothesis 1
      · simpa [List.countP_cons, leftHead, rightHead, Nat.add_assoc,
          Nat.add_comm, Nat.add_left_comm] using
          Nat.add_le_add_right inductionHypothesis 1
      · have raised := Nat.add_le_add_right inductionHypothesis 1
        exact Nat.le_trans
          (by
            simpa [List.countP_cons, leftHead, rightHead, Nat.add_assoc,
              Nat.add_comm, Nat.add_left_comm] using raised)
          (by
            simp [leftHead, rightHead, Nat.add_comm, Nat.add_left_comm])

private theorem sum_map_add
    {Element : Type uSeed}
    (values : List Element)
    (left right : Element -> Rat) :
    (values.map (fun element => left element + right element)).sum =
      (values.map left).sum + (values.map right).sum := by
  induction values with
  | nil => simp [Rat.zero_add]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, inductionHypothesis]
      ac_rfl

theorem Experiment.probabilityBool_mono
    (experiment : Experiment Outcome)
    {left right : Outcome -> Bool}
    (implication : forall outcome,
      left outcome = true -> right outcome = true) :
    experiment.probabilityBool left <= experiment.probabilityBool right := by
  unfold Experiment.probabilityBool Experiment.countBool
  apply div_le_div_of_le
  · apply Rat.natCast_le_natCast.mpr
    apply List.countP_mono_left
    intro seed _ leftHolds
    exact implication (experiment.outcome seed) leftHolds
  · exact rat_cardinality_pos experiment.support

theorem Mixture.probabilityBool_mono
    {Prefix : Type uSeed}
    (mixture : Mixture Prefix Outcome)
    {left right : Outcome -> Bool}
    (implication : forall outcome,
      left outcome = true -> right outcome = true) :
    mixture.probabilityBool left <= mixture.probabilityBool right := by
  unfold Mixture.probabilityBool
  apply div_le_div_of_le
  · apply sum_map_le_sum_map_local
    intro outer _
    exact (mixture.component outer).probabilityBool_mono implication
  · exact rat_cardinality_pos mixture.prefixes

/-- Finite executable union bound for one uniform experiment. -/
theorem Experiment.probabilityBool_or_le
    (experiment : Experiment Outcome)
    (left right : Outcome -> Bool) :
    experiment.probabilityBool (fun outcome => left outcome || right outcome) <=
      experiment.probabilityBool left + experiment.probabilityBool right := by
  unfold Experiment.probabilityBool Experiment.countBool
  have countBound := countP_or_le experiment.support.values
    (fun seed => left (experiment.outcome seed))
    (fun seed => right (experiment.outcome seed))
  have divided := div_le_div_of_le
    (Rat.natCast_le_natCast.mpr countBound)
    (rat_cardinality_pos experiment.support)
  calc
    _ <= ((experiment.support.values.countP
          (fun seed => left (experiment.outcome seed)) : Nat) : Rat) /
        (experiment.support.cardinality : Rat) +
      ((experiment.support.values.countP
          (fun seed => right (experiment.outcome seed)) : Nat) : Rat) /
        (experiment.support.cardinality : Rat) := by
          simpa [Rat.natCast_add, Rat.div_def, Rat.add_mul] using divided
    _ = _ := rfl

/-- Finite executable union bound averaged over an explicit mixture. -/
theorem Mixture.probabilityBool_or_le
    {Prefix : Type uSeed}
    (mixture : Mixture Prefix Outcome)
    (left right : Outcome -> Bool) :
    mixture.probabilityBool (fun outcome => left outcome || right outcome) <=
      mixture.probabilityBool left + mixture.probabilityBool right := by
  unfold Mixture.probabilityBool
  have sumBound :
      (mixture.prefixes.values.map (fun outer =>
        (mixture.component outer).probabilityBool
          (fun outcome => left outcome || right outcome))).sum <=
      (mixture.prefixes.values.map (fun outer =>
        (mixture.component outer).probabilityBool left +
          (mixture.component outer).probabilityBool right)).sum := by
    apply sum_map_le_sum_map_local
    intro outer _
    exact (mixture.component outer).probabilityBool_or_le left right
  have divided := div_le_div_of_le sumBound
    (rat_cardinality_pos mixture.prefixes)
  rw [sum_map_add] at divided
  simpa [Rat.div_def, Rat.add_mul] using divided

/-- Boolean finite-event version of subtractive bad-event accounting. -/
theorem Experiment.probabilityBool_sub_le_of_cover
    (experiment : Experiment Outcome)
    (success extracted bad : Outcome -> Bool)
    (error : Rat)
    (cover : forall outcome, success outcome = true ->
      extracted outcome = true \/ bad outcome = true)
    (badBound : experiment.probabilityBool bad <= error) :
    experiment.probabilityBool success - error <=
      experiment.probabilityBool extracted := by
  have successToUnion : forall outcome,
      success outcome = true ->
        (extracted outcome || bad outcome) = true := by
    intro outcome successHolds
    rcases cover outcome successHolds with extractedHolds | badHolds
    · simp [extractedHolds]
    · simp [badHolds]
  have successBelowUnion := experiment.probabilityBool_mono successToUnion
  have unionBelowAdd := experiment.probabilityBool_or_le extracted bad
  have addBelowBudget :
      experiment.probabilityBool extracted + experiment.probabilityBool bad <=
        experiment.probabilityBool extracted + error :=
    (Rat.add_le_add_left
      (c := experiment.probabilityBool extracted)).mpr badBound
  have successBelowBudget := Rat.le_trans successBelowUnion
    (Rat.le_trans unionBelowAdd addBelowBudget)
  have shifted := (Rat.add_le_add_right (c := -error)).mpr successBelowBudget
  calc
    experiment.probabilityBool success - error =
        experiment.probabilityBool success + -error :=
      Rat.sub_eq_add_neg _ _
    _ <= (experiment.probabilityBool extracted + error) + -error := shifted
    _ = experiment.probabilityBool extracted := by
      rw [Rat.add_assoc, Rat.add_neg_cancel, Rat.add_zero]

/-- Boolean finite-event subtractive accounting for an explicit mixture. -/
theorem Mixture.probabilityBool_sub_le_of_cover
    {Prefix : Type uSeed}
    (mixture : Mixture Prefix Outcome)
    (success extracted bad : Outcome -> Bool)
    (error : Rat)
    (cover : forall outcome, success outcome = true ->
      extracted outcome = true \/ bad outcome = true)
    (badBound : mixture.probabilityBool bad <= error) :
    mixture.probabilityBool success - error <=
      mixture.probabilityBool extracted := by
  have successToUnion : forall outcome,
      success outcome = true ->
        (extracted outcome || bad outcome) = true := by
    intro outcome successHolds
    rcases cover outcome successHolds with extractedHolds | badHolds
    · simp [extractedHolds]
    · simp [badHolds]
  have successBelowUnion := mixture.probabilityBool_mono successToUnion
  have unionBelowAdd := mixture.probabilityBool_or_le extracted bad
  have addBelowBudget :
      mixture.probabilityBool extracted + mixture.probabilityBool bad <=
        mixture.probabilityBool extracted + error :=
    (Rat.add_le_add_left
      (c := mixture.probabilityBool extracted)).mpr badBound
  have successBelowBudget := Rat.le_trans successBelowUnion
    (Rat.le_trans unionBelowAdd addBelowBudget)
  have shifted := (Rat.add_le_add_right (c := -error)).mpr successBelowBudget
  calc
    mixture.probabilityBool success - error =
        mixture.probabilityBool success + -error :=
      Rat.sub_eq_add_neg _ _
    _ <= (mixture.probabilityBool extracted + error) + -error := shifted
    _ = mixture.probabilityBool extracted := by
      rw [Rat.add_assoc, Rat.add_neg_cancel, Rat.add_zero]

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
