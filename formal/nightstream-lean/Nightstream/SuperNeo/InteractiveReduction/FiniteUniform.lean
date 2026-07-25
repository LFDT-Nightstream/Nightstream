import Init.Data.List.Count
import Init.Data.Rat
import Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

/-!
Finite-uniform probability and exact cost accounting for interactive reductions.

Owns: a concrete rational probability scale, duplicate-free finite seed
supports, executable Boolean event counting, finite pushforwards, uniform
finite mixtures, and exact expected-cost inequalities.

Does not own: a protocol, a transcript, an extractor, polynomial-time
complexity, Fiat--Shamir, Rust, R1CS, or a coordinate-forking theorem.

Emits constraints: no.

The support is a list of distinct *seeds*.  An experiment's output map may be
many-to-one; consequently a pushforward never deduplicates outputs or changes
their probability mass.  Arbitrary-`Prop` probabilities are noncomputable
adapters required by `Paper.ProbabilityExperiment`.  The Boolean path and all
cost accounting remain executable.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uSeed uOutcome uOtherOutcome uMapped uResult uPrefix uComponentSeed uQuery

variable {Seed : Type uSeed}
variable {Outcome : Type uOutcome}
variable {OtherOutcome : Type uOtherOutcome}
variable {Mapped : Type uMapped}
variable {Result : Type uResult}
variable {Prefix : Type uPrefix}
variable {Query : Type uQuery}

/-- An explicit, nonempty, duplicate-free finite seed support. -/
structure Support (Seed : Type uSeed) where
  values : List Seed
  nodup : values.Nodup
  nonempty : values ≠ []

/-- Number of equally likely seeds. -/
def Support.cardinality (support : Support Seed) : Nat :=
  support.values.length

theorem Support.cardinality_pos (support : Support Seed) :
    0 < support.cardinality := by
  exact List.length_pos_iff.mpr support.nonempty

private theorem Support.cardinality_rat_pos (support : Support Seed) :
    0 < (support.cardinality : Rat) := by
  exact Rat.natCast_pos.mpr support.cardinality_pos

private theorem Support.cardinality_rat_ne_zero (support : Support Seed) :
    (support.cardinality : Rat) ≠ 0 :=
  Rat.ne_of_gt support.cardinality_rat_pos

/-- The concrete rational probability scale used by finite experiments. -/
def scale : ProbabilityScale Rat where
  zero := 0
  one := 1
  add := (fun left right => left + right)
  subtract := (fun left right => left - right)
  le := (fun left right => left <= right)
  le_refl := fun weight => Rat.le_refl
  le_trans := Rat.le_trans
  subtract_zero := by
    intro weight
    simp [Rat.sub_eq_add_neg, Rat.add_zero]

/-- Standard ordered-additive laws for the concrete rational scale. -/
def scaleLaws : ScaleLaws scale where
  add_mono := by
    intro left lower right upper leftBound rightBound
    exact Rat.le_trans
      ((Rat.add_le_add_right (c := right)).mpr leftBound)
      ((Rat.add_le_add_left (c := lower)).mpr rightBound)
  subtract_le_of_le_add := by
    intro probability good error bound
    have shifted :=
      (Rat.add_le_add_right (c := -error)).mpr bound
    change probability - error <= good
    calc
      probability - error = probability + -error :=
        Rat.sub_eq_add_neg _ _
      _ <= (good + error) + -error := shifted
      _ = good := by
        rw [Rat.add_assoc, Rat.add_neg_cancel, Rat.add_zero]

/-- Exact rational ratio.  Callers must separately prove a positive
denominator when they use order or cancellation laws. -/
def ratio (numerator denominator : Nat) : Rat :=
  (numerator : Rat) / (denominator : Rat)

/-- A finite experiment is a uniform seed followed by a deterministic output
map.  Keeping the seed support explicit preserves multiplicity under maps. -/
structure Experiment (Outcome : Type uOutcome) where
  Seed : Type uSeed
  support : Support Seed
  outcome : Seed -> Outcome

/-- The uniform experiment on a support itself. -/
def Support.uniform (support : Support Seed) : Experiment Seed where
  Seed := Seed
  support := support
  outcome := id

/-- Deterministic pushforward.  This does not deduplicate mapped outputs. -/
def Experiment.map
    (experiment : Experiment Outcome)
    (mapping : Outcome -> Mapped) : Experiment Mapped where
  Seed := experiment.Seed
  support := experiment.support
  outcome := fun seed => mapping (experiment.outcome seed)

theorem Experiment.map_id
    (experiment : Experiment Outcome) :
    experiment.map id = experiment := by
  cases experiment
  rfl

theorem Experiment.map_map
    (experiment : Experiment Outcome)
    (first : Outcome -> Mapped)
    (second : Mapped -> Result) :
    (experiment.map first).map second =
      experiment.map (fun outcome => second (first outcome)) := by
  cases experiment
  rfl

/-- Executable event count over the uniform seed support. -/
def Experiment.countBool
    (experiment : Experiment Outcome)
    (event : Outcome -> Bool) : Nat :=
  experiment.support.values.countP
    (fun seed => event (experiment.outcome seed))

/-- Executable rational probability of a Boolean event. -/
def Experiment.probabilityBool
    (experiment : Experiment Outcome)
    (event : Outcome -> Bool) : Rat :=
  (experiment.countBool event : Rat) /
    (experiment.support.cardinality : Rat)

private noncomputable def propTest
    (event : Outcome -> Prop) (outcome : Outcome) : Bool :=
  @ite Bool (event outcome) (Classical.propDecidable _) true false

@[simp] private theorem propTest_eq_true
    (event : Outcome -> Prop) (outcome : Outcome) :
    propTest event outcome = true <-> event outcome := by
  simp [propTest]

@[simp] private theorem propTest_bool
    (event : Outcome -> Bool) (outcome : Outcome) :
    propTest (fun value => event value = true) outcome = event outcome := by
  cases h : event outcome <;> simp [propTest, h]

/-- Mathematical event count.  Noncomputability is confined to deciding an
arbitrary proposition; concrete verifier events should use `countBool`. -/
noncomputable def Experiment.count
    (experiment : Experiment Outcome)
    (event : Outcome -> Prop) : Nat :=
  experiment.countBool (propTest event)

/-- Exact rational probability of an arbitrary mathematical event. -/
noncomputable def Experiment.probability
    (experiment : Experiment Outcome)
    (event : Outcome -> Prop) : Rat :=
  (experiment.count event : Rat) /
    (experiment.support.cardinality : Rat)

theorem Experiment.probability_bool_event
    (experiment : Experiment Outcome)
    (event : Outcome -> Bool) :
    experiment.probability (fun outcome => event outcome = true) =
      experiment.probabilityBool event := by
  have predicatesEqual :
      (fun seed => propTest (fun outcome => event outcome = true)
        (experiment.outcome seed)) =
      (fun seed => event (experiment.outcome seed)) := by
    funext seed
    exact propTest_bool event (experiment.outcome seed)
  unfold Experiment.probability Experiment.count
    Experiment.probabilityBool Experiment.countBool
  rw [predicatesEqual]

theorem Experiment.count_mono
    (experiment : Experiment Outcome)
    {left right : Outcome -> Prop}
    (implication : forall outcome, left outcome -> right outcome) :
    experiment.count left <= experiment.count right := by
  unfold Experiment.count Experiment.countBool
  apply List.countP_mono_left
  intro seed _ leftHolds
  have leftProp : left (experiment.outcome seed) := by
    simpa only [propTest_eq_true] using leftHolds
  have rightProp := implication (experiment.outcome seed) leftProp
  simpa only [propTest_eq_true] using rightProp

theorem div_le_div_of_le
    {left right denominator : Rat}
    (ordered : left <= right)
    (denominatorPos : 0 < denominator) :
    left / denominator <= right / denominator := by
  simp only [Rat.div_def]
  exact Rat.mul_le_mul_of_nonneg_right ordered
    (Rat.le_of_lt (Rat.inv_pos.mpr denominatorPos))

/-- Divide an upper bound by a positive denominator without hiding the
cross-multiplication step. -/
theorem le_div_iff_of_pos
    {left right denominator : Rat}
    (denominatorPos : 0 < denominator) :
    left <= right / denominator <-> left * denominator <= right := by
  constructor
  · intro dividedBound
    apply Rat.not_lt.mp
    intro crossGreater
    have dividedGreater : right / denominator < left :=
      (Rat.div_lt_iff denominatorPos).mpr crossGreater
    exact (Rat.not_lt.mpr dividedBound) dividedGreater
  · intro crossBound
    apply Rat.not_lt.mp
    intro dividedGreater
    have crossGreater : right < left * denominator :=
      (Rat.div_lt_iff denominatorPos).mp dividedGreater
    exact (Rat.not_lt.mpr crossBound) crossGreater

/-- Move a positive denominator across a quotient on the left. -/
theorem div_le_iff_of_pos
    {left right denominator : Rat}
    (denominatorPos : 0 < denominator) :
    left / denominator <= right <-> left <= right * denominator := by
  constructor
  · intro dividedBound
    apply Rat.not_lt.mp
    intro crossGreater
    have dividedGreater : right < left / denominator :=
      (Rat.lt_div_iff denominatorPos).mpr crossGreater
    exact (Rat.not_lt.mpr dividedBound) dividedGreater
  · intro crossBound
    apply Rat.not_lt.mp
    intro dividedGreater
    have crossGreater : right * denominator < left :=
      (Rat.lt_div_iff denominatorPos).mp dividedGreater
    exact (Rat.not_lt.mpr crossBound) crossGreater

theorem Experiment.probability_mono
    (experiment : Experiment Outcome)
    {left right : Outcome -> Prop}
    (implication : forall outcome, left outcome -> right outcome) :
    experiment.probability left <= experiment.probability right := by
  apply div_le_div_of_le
  · exact Rat.natCast_le_natCast.mpr (experiment.count_mono implication)
  · exact experiment.support.cardinality_rat_pos

/-- Finite union bound for arbitrary mathematical predicates. -/
theorem Experiment.probability_or_le
    (experiment : Experiment Outcome)
    (left right : Outcome -> Prop) :
    experiment.probability (fun outcome => left outcome \/ right outcome) <=
      experiment.probability left + experiment.probability right := by
  have countPOr :
      forall (values : List experiment.Seed)
        (leftTest rightTest : experiment.Seed -> Bool),
        values.countP (fun seed => leftTest seed || rightTest seed) <=
          values.countP leftTest + values.countP rightTest := by
    intro values leftTest rightTest
    induction values with
    | nil => simp
    | cons head tail inductionHypothesis =>
        cases leftValue : leftTest head <;>
          cases rightValue : rightTest head
        · simpa [leftValue, rightValue] using inductionHypothesis
        · have shifted := Nat.add_le_add_right inductionHypothesis 1
          simpa [leftValue, rightValue, Nat.add_assoc] using shifted
        · have shifted := Nat.add_le_add_right inductionHypothesis 1
          simpa [leftValue, rightValue, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using shifted
        · simp [leftValue, rightValue]
          omega
  have unionTest :
      (fun seed =>
        propTest (fun outcome => left outcome \/ right outcome)
          (experiment.outcome seed)) =
        (fun seed =>
          propTest left (experiment.outcome seed) ||
            propTest right (experiment.outcome seed)) := by
    funext seed
    by_cases leftHolds : left (experiment.outcome seed) <;>
      by_cases rightHolds : right (experiment.outcome seed) <;>
      simp [propTest, leftHolds, rightHolds]
  have countBound :
      experiment.count (fun outcome => left outcome \/ right outcome) <=
        experiment.count left + experiment.count right := by
    unfold Experiment.count Experiment.countBool
    rw [unionTest]
    exact countPOr experiment.support.values
      (fun seed => propTest left (experiment.outcome seed))
      (fun seed => propTest right (experiment.outcome seed))
  unfold Experiment.probability
  have divided := div_le_div_of_le
    (Rat.natCast_le_natCast.mpr countBound)
    experiment.support.cardinality_rat_pos
  simpa [Rat.natCast_add, Rat.div_def, Rat.add_mul] using divided

theorem Experiment.probability_false
    (experiment : Experiment Outcome) :
    experiment.probability (fun _ => False) = 0 := by
  have countZero : experiment.count (fun _ => False) = 0 := by
    unfold Experiment.count Experiment.countBool
    exact List.countP_eq_zero.mpr (by
      intro seed _
      simp only [propTest_eq_true]
      exact False.elim)
  simp [Experiment.probability, countZero, Rat.div_def]

theorem Experiment.probability_true
    (experiment : Experiment Outcome) :
    experiment.probability (fun _ => True) = 1 := by
  have countAll : experiment.count (fun _ => True) =
      experiment.support.cardinality := by
    unfold Experiment.count Experiment.countBool Support.cardinality
    exact List.countP_eq_length.mpr (by
      intro seed _
      simp only [propTest_eq_true])
  unfold Experiment.probability
  rw [countAll, Rat.div_def]
  exact Rat.mul_inv_cancel _ experiment.support.cardinality_rat_ne_zero

theorem Experiment.map_probability
    (experiment : Experiment Outcome)
    (mapping : Outcome -> Mapped)
    (event : Mapped -> Prop) :
    (experiment.map mapping).probability event =
      experiment.probability (fun outcome => event (mapping outcome)) := by
  rfl

theorem Experiment.map_probabilityBool
    (experiment : Experiment Outcome)
    (mapping : Outcome -> Mapped)
    (event : Mapped -> Bool) :
    (experiment.map mapping).probabilityBool event =
      experiment.probabilityBool (fun outcome => event (mapping outcome)) := by
  rfl

/-- Concrete implementation of the paper's abstract probability experiment. -/
noncomputable def Experiment.toProbabilityExperiment
    (experiment : Experiment Outcome) :
    ProbabilityExperiment scale Outcome where
  probability := experiment.probability
  monotone := by
    intro left right implication
    exact experiment.probability_mono implication

/-- Standard union-bound adapter for the concrete finite experiment. -/
noncomputable def Experiment.toProbabilityUnionBound
    (experiment : Experiment Outcome) :
    UnionBound experiment.toProbabilityExperiment where
  unionBound := experiment.probability_or_le

private theorem sum_map_le_sum_map
    (values : List Prefix)
    (left right : Prefix -> Rat)
    (ordered : forall value, value ∈ values -> left value <= right value) :
    (values.map left).sum <= (values.map right).sum := by
  induction values with
  | nil => exact Rat.le_refl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons]
      have headOrdered := ordered head (by simp)
      have tailOrdered : forall value, value ∈ tail ->
          left value <= right value := by
        intro value member
        exact ordered value (by simp [member])
      exact Rat.le_trans
        ((Rat.add_le_add_right (c := (tail.map left).sum)).mpr headOrdered)
        ((Rat.add_le_add_left (c := right head)).mpr
          (inductionHypothesis tailOrdered))

private theorem sum_map_constant
    (values : List Prefix)
    (constant : Rat) :
    (values.map (fun _ => constant)).sum =
      (values.length : Rat) * constant := by
  induction values with
  | nil => simp
  | cons _ tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        Rat.natCast_add, inductionHypothesis, Rat.natCast_ofNat,
        Rat.add_mul, Rat.one_mul]
      rw [Rat.add_comm]

private theorem add_sub_add_sub
    (leftValue lossLeft rightValue lossRight : Rat) :
    (leftValue - lossLeft) + (rightValue - lossRight) =
      (leftValue + rightValue) - (lossLeft + lossRight) := by
  simp [Rat.sub_eq_add_neg, Rat.neg_add, Rat.add_assoc,
    Rat.add_left_comm]

private theorem sum_map_sub_constant
    (values : List Prefix) (value : Prefix -> Rat) (loss : Rat) :
    (values.map (fun outer => value outer - loss)).sum =
      (values.map value).sum - (values.length : Rat) * loss := by
  induction values with
  | nil =>
      change (0 : Rat) = 0 - 0 * loss
      rw [Rat.zero_mul, Rat.sub_eq_add_neg, Rat.neg_zero]
      exact (Rat.add_zero 0).symm
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      calc
        (value head - loss) +
              (tail.map (fun outer => value outer - loss)).sum =
            (value head - loss) +
              ((tail.map value).sum - (tail.length : Rat) * loss) := by
                rw [inductionHypothesis]
        _ = (value head + (tail.map value).sum) -
              (loss + (tail.length : Rat) * loss) :=
                add_sub_add_sub _ _ _ _
        _ = (value head + (tail.map value).sum) -
              (↑tail.length.succ : Rat) * loss := by
                congr 1
                rw [Nat.succ_eq_add_one, Rat.natCast_add, Rat.add_mul]
                change loss + (tail.length : Rat) * loss =
                  (tail.length : Rat) * loss + (1 : Rat) * loss
                rw [Rat.one_mul, Rat.add_comm]

private theorem div_sub_distrib
    (left right denominator : Rat) :
    (left - right) / denominator =
      left / denominator - right / denominator := by
  simp [Rat.div_def, Rat.sub_eq_add_neg, Rat.add_mul, Rat.neg_mul]

private theorem average_sub_constant
    (values : List Prefix)
    (nonempty : values ≠ [])
    (value : Prefix -> Rat)
    (loss : Rat) :
    (values.map (fun outer => value outer - loss)).sum /
        (values.length : Rat) =
      (values.map value).sum / (values.length : Rat) - loss := by
  have lengthPos : 0 < values.length := List.length_pos_iff.mpr nonempty
  have denominatorNe : (values.length : Rat) ≠ 0 :=
    Rat.ne_of_gt (Rat.natCast_pos.mpr lengthPos)
  rw [sum_map_sub_constant, div_sub_distrib]
  have cancel : ((values.length : Rat) * loss) /
      (values.length : Rat) = loss := by
    rw [Rat.mul_comm]
    exact Rat.mul_div_cancel denominatorNe
  rw [cancel]

/-- Uniform finite mixture over explicit prefixes.  Components may use
different seed types, but those seed types inhabit one shared universe. -/
structure Mixture (Prefix : Type uPrefix) (Outcome : Type uOutcome) where
  prefixes : Support Prefix
  component : Prefix -> Experiment.{uComponentSeed, uOutcome} Outcome

/-- Executable Boolean-event probability for a uniform outer mixture. -/
def Mixture.probabilityBool
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Bool) : Rat :=
  (mixture.prefixes.values.map
      (fun outer => (mixture.component outer).probabilityBool event)).sum /
    (mixture.prefixes.cardinality : Rat)

/-- Mathematical event probability for a uniform outer mixture. -/
noncomputable def Mixture.probability
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Prop) : Rat :=
  (mixture.prefixes.values.map
      (fun outer => (mixture.component outer).probability event)).sum /
    (mixture.prefixes.cardinality : Rat)

/-- Component-wise deterministic pushforward. -/
def Mixture.map
    (mixture : Mixture Prefix Outcome)
    (mapping : Outcome -> Mapped) :
    Mixture Prefix Mapped where
  prefixes := mixture.prefixes
  component := fun outer => (mixture.component outer).map mapping

theorem Mixture.map_map
    (mixture : Mixture Prefix Outcome)
    (first : Outcome -> Mapped)
    (second : Mapped -> Result) :
    (mixture.map first).map second =
      mixture.map (fun outcome => second (first outcome)) := by
  cases mixture
  rfl

theorem Mixture.map_probability
    (mixture : Mixture Prefix Outcome)
    (mapping : Outcome -> Mapped)
    (event : Mapped -> Prop) :
    (mixture.map mapping).probability event =
      mixture.probability (fun outcome => event (mapping outcome)) := by
  rfl

theorem Mixture.map_probabilityBool
    (mixture : Mixture Prefix Outcome)
    (mapping : Outcome -> Mapped)
    (event : Mapped -> Bool) :
    (mixture.map mapping).probabilityBool event =
      mixture.probabilityBool (fun outcome => event (mapping outcome)) := by
  rfl

theorem Mixture.probability_bool_event
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Bool) :
    mixture.probability (fun outcome => event outcome = true) =
      mixture.probabilityBool event := by
  have componentProbabilitiesEqual :
      (fun outer => (mixture.component outer).probability
        (fun outcome => event outcome = true)) =
      (fun outer => (mixture.component outer).probabilityBool event) := by
    funext outer
    exact (mixture.component outer).probability_bool_event event
  unfold Mixture.probability Mixture.probabilityBool
  rw [componentProbabilitiesEqual]

theorem Mixture.probability_mono
    (mixture : Mixture Prefix Outcome)
    {left right : Outcome -> Prop}
    (implication : forall outcome, left outcome -> right outcome) :
    mixture.probability left <= mixture.probability right := by
  unfold Mixture.probability
  apply div_le_div_of_le
  · apply sum_map_le_sum_map
    intro outer _
    exact (mixture.component outer).probability_mono implication
  · exact mixture.prefixes.cardinality_rat_pos

theorem Mixture.probability_false
    (mixture : Mixture Prefix Outcome) :
    mixture.probability (fun _ => False) = 0 := by
  unfold Mixture.probability
  have componentZero :
      (fun outer =>
        (mixture.component outer).probability (fun _ => False)) =
        (fun _ => (0 : Rat)) := by
    funext outer
    exact (mixture.component outer).probability_false
  rw [componentZero]
  have sumZero :
      (mixture.prefixes.values.map (fun _ => (0 : Rat))).sum = 0 := by
    induction mixture.prefixes.values with
    | nil => rfl
    | cons _ tail inductionHypothesis =>
        simp only [List.map_cons, List.sum_cons, inductionHypothesis]
        exact Rat.zero_add 0
  rw [sumZero]
  simp only [Rat.div_def]
  exact Rat.zero_mul _

/-- Finite union bound for arbitrary predicates over a uniform outer
mixture. -/
theorem Mixture.probability_or_le
    (mixture : Mixture Prefix Outcome)
    (left right : Outcome -> Prop) :
    mixture.probability (fun outcome => left outcome \/ right outcome) <=
      mixture.probability left + mixture.probability right := by
  let values := mixture.prefixes.values
  let unionProbability : Prefix -> Rat := fun outer =>
    (mixture.component outer).probability
      (fun outcome => left outcome \/ right outcome)
  let leftProbability : Prefix -> Rat := fun outer =>
    (mixture.component outer).probability left
  let rightProbability : Prefix -> Rat := fun outer =>
    (mixture.component outer).probability right
  have componentBound :
      (values.map unionProbability).sum <=
        (values.map (fun outer =>
          leftProbability outer + rightProbability outer)).sum := by
    apply sum_map_le_sum_map
    intro outer member
    exact (mixture.component outer).probability_or_le left right
  have sumAdd :
      (values.map (fun outer =>
          leftProbability outer + rightProbability outer)).sum =
        (values.map leftProbability).sum +
          (values.map rightProbability).sum := by
    induction values with
    | nil => exact (Rat.zero_add 0).symm
    | cons head tail inductionHypothesis =>
        simp only [List.map_cons, List.sum_cons]
        rw [inductionHypothesis]
        simp [Rat.add_assoc, Rat.add_left_comm]
  have denominatorPos :
      0 < (mixture.prefixes.cardinality : Rat) :=
    mixture.prefixes.cardinality_rat_pos
  change
    (values.map unionProbability).sum /
        (mixture.prefixes.cardinality : Rat) <=
      (values.map leftProbability).sum /
          (mixture.prefixes.cardinality : Rat) +
        (values.map rightProbability).sum /
          (mixture.prefixes.cardinality : Rat)
  calc
    _ <=
        ((values.map leftProbability).sum +
            (values.map rightProbability).sum) /
          (mixture.prefixes.cardinality : Rat) := by
      apply div_le_div_of_le
      · simpa only [sumAdd] using componentBound
      · exact denominatorPos
    _ = _ := by
      simp [Rat.div_def, Rat.add_mul]

/-- Pointwise conditional extraction bounds average without multiplying the
loss by the number of prefixes. -/
theorem Mixture.loss_le_of_components
    (mixture : Mixture Prefix Outcome)
    (success fork : Outcome -> Prop)
    (loss : Rat)
    (componentBound : forall outer, outer ∈ mixture.prefixes.values ->
      (mixture.component outer).probability success - loss <=
        (mixture.component outer).probability fork) :
    mixture.probability success - loss <= mixture.probability fork := by
  unfold Mixture.probability
  simp only [Support.cardinality]
  rw [← average_sub_constant mixture.prefixes.values
    mixture.prefixes.nonempty
    (fun outer => (mixture.component outer).probability success) loss]
  apply div_le_div_of_le
  · apply sum_map_le_sum_map
    intro outer member
    exact componentBound outer member
  · exact mixture.prefixes.cardinality_rat_pos

/-- Pointwise bounds between two experiments over the same outer support
average with the same loss.  The component outcome and seed types may differ;
only the authoritative prefix distribution must be identical. -/
theorem Mixture.loss_le_of_component_pairs
    (left : Mixture Prefix Outcome)
    (right : Mixture Prefix OtherOutcome)
    (samePrefixes : left.prefixes = right.prefixes)
    (leftEvent : Outcome -> Prop)
    (rightEvent : OtherOutcome -> Prop)
    (loss : Rat)
    (componentBound : forall outer, outer ∈ left.prefixes.values ->
      (left.component outer).probability leftEvent - loss <=
        (right.component outer).probability rightEvent) :
    left.probability leftEvent - loss <= right.probability rightEvent := by
  cases left with
  | mk leftPrefixes leftComponent =>
      cases right with
      | mk rightPrefixes rightComponent =>
          dsimp only at samePrefixes
          subst rightPrefixes
          unfold Mixture.probability
          simp only [Support.cardinality]
          rw [← average_sub_constant leftPrefixes.values
            leftPrefixes.nonempty
            (fun outer => (leftComponent outer).probability leftEvent) loss]
          apply div_le_div_of_le
          · apply sum_map_le_sum_map
            intro outer member
            exact componentBound outer member
          · exact leftPrefixes.cardinality_rat_pos

/-- A common pointwise security bound survives a uniform outer mixture.

This is the witness-uniqueness counterpart of `loss_le_of_components`: if
every fixed-prefix disagreement experiment is bounded by the same relaxed-
binding error, averaging over prefixes does not multiply that error. -/
theorem Mixture.probability_le_of_components
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Prop)
    (bound : Rat)
    (componentBound : forall outer, outer ∈ mixture.prefixes.values ->
      (mixture.component outer).probability event <= bound) :
    mixture.probability event <= bound := by
  let componentProbability : Prefix -> Rat := fun outer =>
    (mixture.component outer).probability event
  have sumBound :
      (mixture.prefixes.values.map componentProbability).sum <=
        (mixture.prefixes.values.map (fun _ => bound)).sum := by
    apply sum_map_le_sum_map
    intro outer member
    exact componentBound outer member
  have divided := div_le_div_of_le sumBound
    mixture.prefixes.cardinality_rat_pos
  unfold Mixture.probability
  change
    (mixture.prefixes.values.map componentProbability).sum /
        (mixture.prefixes.values.length : Rat) <= bound
  unfold Support.cardinality at divided
  have lengthPositive : 0 < (mixture.prefixes.values.length : Rat) := by
    simpa only [Support.cardinality] using
      mixture.prefixes.cardinality_rat_pos
  rw [sum_map_constant,
    Rat.mul_comm (mixture.prefixes.values.length : Rat) bound,
    Rat.mul_div_cancel
      (Rat.ne_of_gt lengthPositive)] at divided
  exact divided

/-- Concrete implementation of the paper's abstract probability experiment
for a finite conditional mixture. -/
noncomputable def Mixture.toProbabilityExperiment
    (mixture : Mixture Prefix Outcome) :
    ProbabilityExperiment scale Outcome where
  probability := mixture.probability
  monotone := by
    intro left right implication
    exact mixture.probability_mono implication

/-- Standard union-bound adapter for a finite conditional mixture. -/
noncomputable def Mixture.toProbabilityUnionBound
    (mixture : Mixture Prefix Outcome) :
    UnionBound mixture.toProbabilityExperiment where
  unionBound := mixture.probability_or_le

/-- Sum of a natural-valued cost over every equally likely seed. -/
def Experiment.totalCost
    (experiment : Experiment Outcome)
    (cost : experiment.Seed -> Nat) : Nat :=
  (experiment.support.values.map cost).sum

/-- Exact expected cost as a rational number. -/
def Experiment.expectedCost
    (experiment : Experiment Outcome)
    (cost : experiment.Seed -> Nat) : Rat :=
  (experiment.totalCost cost : Rat) /
    (experiment.support.cardinality : Rat)

/-- Cross-multiplied natural-number expected-cost bound.  This definition is
executable and makes the denominator explicit. -/
def Experiment.ExpectedCostAtMost
    (experiment : Experiment Outcome)
    (cost : experiment.Seed -> Nat)
    (bound : Nat) : Prop :=
  experiment.totalCost cost <= bound * experiment.support.cardinality

theorem Experiment.expectedCost_le_iff_totalCost_le
    (experiment : Experiment Outcome)
    (cost : experiment.Seed -> Nat)
    (bound : Nat) :
    experiment.expectedCost cost <= (bound : Rat) <->
      (experiment.totalCost cost : Rat) <=
        (bound : Rat) * (experiment.support.cardinality : Rat) := by
  have denominatorPos := experiment.support.cardinality_rat_pos
  constructor
  · intro expectedBound
    apply Rat.not_lt.mp
    intro crossGreater
    have expectedGreater : (bound : Rat) < experiment.expectedCost cost := by
      exact (Rat.lt_div_iff denominatorPos).mpr crossGreater
    exact (Rat.not_lt.mpr expectedBound) expectedGreater
  · intro crossBound
    apply Rat.not_lt.mp
    intro expectedGreater
    have crossGreater :
        (bound : Rat) * (experiment.support.cardinality : Rat) <
          (experiment.totalCost cost : Rat) :=
      (Rat.lt_div_iff denominatorPos).mp expectedGreater
    exact (Rat.not_lt.mpr crossBound) crossGreater

theorem Experiment.expectedCost_le_iff_expectedCostAtMost
    (experiment : Experiment Outcome)
    (cost : experiment.Seed -> Nat)
    (bound : Nat) :
    experiment.expectedCost cost <= (bound : Rat) <->
      experiment.ExpectedCostAtMost cost bound := by
  rw [experiment.expectedCost_le_iff_totalCost_le]
  unfold Experiment.ExpectedCostAtMost
  rw [← Rat.natCast_mul, Rat.natCast_le_natCast]

/-- Exact expected-query bound obtained by charging the length of the query
trace returned for each seed. -/
def Experiment.ExpectedQueriesAtMost
    (experiment : Experiment Outcome)
    (trace : experiment.Seed -> List Query)
    (bound : Nat) : Prop :=
  experiment.ExpectedCostAtMost (fun seed => (trace seed).length) bound

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
