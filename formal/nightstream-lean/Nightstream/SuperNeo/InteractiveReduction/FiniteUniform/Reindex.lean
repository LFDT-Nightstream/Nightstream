import Init.Data.List.Perm
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Product

/-!
Reindexing kernels for finite-uniform experiments.

Owns: invariance of Boolean and propositional event probabilities under an
explicit permutation of mapped seed supports, and equivalence between a
uniform outer mixture with one shared inner support and the corresponding
uniform Cartesian-product experiment.

Does not own: a protocol-specific seed map, a protocol event, Fiat--Shamir,
Rust, R1CS, artifacts, or costs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uLeftSeed uRightSeed uLeftOutcome uRightOutcome
  uPrefix uInner uOutcome

variable {LeftOutcome : Type uLeftOutcome}
variable {RightOutcome : Type uRightOutcome}
variable {LeftSeed : Type uLeftSeed}
variable {RightSeed : Type uRightSeed}
variable {Prefix : Type uPrefix}
variable {Inner : Type uInner}
variable {Outcome : Type uOutcome}

/-- A two-sided seed reindex that preserves support membership maps one
duplicate-free support to a permutation of the other. -/
theorem Support.map_values_perm_of_inverse
    (left : Support LeftSeed)
    (right : Support RightSeed)
    (forward : LeftSeed -> RightSeed)
    (backward : RightSeed -> LeftSeed)
    (leftInverse : forall seed, backward (forward seed) = seed)
    (rightInverse : forall seed, forward (backward seed) = seed)
    (membership : forall seed,
      seed ∈ left.values <-> forward seed ∈ right.values) :
    (left.values.map forward).Perm right.values := by
  classical
  have forwardInjective : Function.Injective forward := by
    intro first second equal
    simpa only [leftInverse] using congrArg backward equal
  have mappedNodup : (left.values.map forward).Nodup :=
    left.nodup.map forward (by
      intro first second distinct equal
      exact distinct (forwardInjective equal))
  have mappedMembership : forall seed,
      seed ∈ left.values.map forward <-> seed ∈ right.values := by
    intro seed
    constructor
    · intro member
      rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
      exact (membership source).mp sourceMember
    · intro member
      apply List.mem_map.mpr
      refine ⟨backward seed, ?_, rightInverse seed⟩
      apply (membership (backward seed)).mpr
      simpa only [rightInverse] using member
  apply List.perm_iff_count.mpr
  intro seed
  rw [mappedNodup.count, right.nodup.count]
  by_cases mappedMember : seed ∈ left.values.map forward
  · have rightMember := (mappedMembership seed).mp mappedMember
    simp [mappedMember, rightMember]
  · have rightNotMember : seed ∉ right.values := by
      intro rightMember
      exact mappedMember ((mappedMembership seed).mpr rightMember)
    simp [mappedMember, rightNotMember]

/-- A mapped permutation of the finite seed support, together with pointwise
agreement of the Boolean events, preserves the exact event probability. -/
theorem Experiment.probabilityBool_eq_of_reindex
    (left : Experiment LeftOutcome)
    (right : Experiment RightOutcome)
    (forward : left.Seed -> right.Seed)
    (supportPermutation :
      (left.support.values.map forward).Perm right.support.values)
    (leftEvent : LeftOutcome -> Bool)
    (rightEvent : RightOutcome -> Bool)
    (eventAgreement : forall seed, seed ∈ left.support.values ->
      leftEvent (left.outcome seed) =
        rightEvent (right.outcome (forward seed))) :
    left.probabilityBool leftEvent =
      right.probabilityBool rightEvent := by
  have cardinalityAgreement :
      left.support.cardinality = right.support.cardinality := by
    unfold Support.cardinality
    calc
      left.support.values.length =
          (left.support.values.map forward).length := by
            simp
      _ = right.support.values.length := supportPermutation.length_eq
  have countAgreement :
      left.countBool leftEvent = right.countBool rightEvent := by
    unfold Experiment.countBool
    calc
      left.support.values.countP
            (fun seed => leftEvent (left.outcome seed)) =
          left.support.values.countP
            (fun seed => rightEvent (right.outcome (forward seed))) := by
              apply List.countP_congr
              intro seed member
              simpa [eventAgreement seed member]
      _ = (left.support.values.map forward).countP
            (fun seed => rightEvent (right.outcome seed)) := by
              simpa only [Function.comp_apply] using
                (List.countP_map
                  (l := left.support.values)
                  (f := forward)
                  (p := fun seed => rightEvent (right.outcome seed))).symm
      _ = right.support.values.countP
            (fun seed => rightEvent (right.outcome seed)) :=
              supportPermutation.countP_eq _
  unfold Experiment.probabilityBool
  rw [countAgreement, cardinalityAgreement]

/-- Propositional-event form of `probabilityBool_eq_of_reindex`.  The only
noncomputability is the local proposition-to-Boolean adapter already used by
the finite-uniform probability model. -/
theorem Experiment.probability_eq_of_reindex
    (left : Experiment LeftOutcome)
    (right : Experiment RightOutcome)
    (forward : left.Seed -> right.Seed)
    (supportPermutation :
      (left.support.values.map forward).Perm right.support.values)
    (leftEvent : LeftOutcome -> Prop)
    (rightEvent : RightOutcome -> Prop)
    (eventAgreement : forall seed, seed ∈ left.support.values ->
      (leftEvent (left.outcome seed) <->
        rightEvent (right.outcome (forward seed)))) :
    left.probability leftEvent =
      right.probability rightEvent := by
  classical
  let leftTest : LeftOutcome -> Bool := fun outcome =>
    if leftEvent outcome then true else false
  let rightTest : RightOutcome -> Bool := fun outcome =>
    if rightEvent outcome then true else false
  have leftEventEq :
      leftEvent = (fun outcome => leftTest outcome = true) := by
    funext outcome
    apply propext
    simp [leftTest]
  have rightEventEq :
      rightEvent = (fun outcome => rightTest outcome = true) := by
    funext outcome
    apply propext
    simp [rightTest]
  have testAgreement : forall seed, seed ∈ left.support.values ->
      leftTest (left.outcome seed) =
        rightTest (right.outcome (forward seed)) := by
    intro seed member
    have agreement := eventAgreement seed member
    by_cases leftHolds : leftEvent (left.outcome seed)
    · have rightHolds :
          rightEvent (right.outcome (forward seed)) :=
        agreement.mp leftHolds
      simp [leftTest, rightTest, leftHolds, rightHolds]
    · have rightFails :
          ¬rightEvent (right.outcome (forward seed)) := by
        intro rightHolds
        exact leftHolds (agreement.mpr rightHolds)
      simp [leftTest, rightTest, leftHolds, rightFails]
  calc
    left.probability leftEvent =
        left.probability (fun outcome => leftTest outcome = true) := by
          rw [leftEventEq]
    _ = left.probabilityBool leftTest :=
      left.probability_bool_event leftTest
    _ = right.probabilityBool rightTest :=
      Experiment.probabilityBool_eq_of_reindex left right forward
        supportPermutation leftTest rightTest testAgreement
    _ = right.probability
          (fun outcome => rightTest outcome = true) :=
      (right.probability_bool_event rightTest).symm
    _ = right.probability rightEvent := by
      rw [rightEventEq]

private theorem sum_map_div
    {Index : Type uPrefix}
    (values : List Index)
    (value : Index -> Rat)
    (denominator : Rat) :
    (values.map (fun index => value index / denominator)).sum =
      (values.map value).sum / denominator := by
  induction values with
  | nil =>
      simp [Rat.div_def]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons]
      rw [inductionHypothesis]
      simp [Rat.div_def, Rat.add_mul]

private theorem natCast_sum (values : List Nat) :
    (values.sum : Rat) =
      (values.map fun value : Nat => (value : Rat)).sum := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.sum_cons, List.map_cons]
      rw [Rat.natCast_add, inductionHypothesis]

/-- Averaging over a uniform outer support and then a shared uniform inner
support is exactly the uniform experiment on their lexicographic Cartesian
product.  Multiplicity is preserved; neither side deduplicates outcomes. -/
theorem Mixture.sharedSupport_probabilityBool_eq_product
    (prefixes : Support Prefix)
    (inner : Support Inner)
    (outcome : Prefix -> Inner -> Outcome)
    (event : Outcome -> Bool) :
    ({ prefixes := prefixes
       component := fun outer =>
        { Seed := Inner
          support := inner
          outcome := outcome outer } } :
      Mixture Prefix Outcome).probabilityBool event =
    ({ Seed := Prefix × Inner
       support := prefixes.product inner
       outcome := fun seed => outcome seed.1 seed.2 } :
      Experiment Outcome).probabilityBool event := by
  let componentCount : Prefix -> Nat := fun outer =>
    inner.values.countP (fun seed => event (outcome outer seed))
  have productCount :
      (prefixes.product inner).values.countP
          (fun seed => event (outcome seed.1 seed.2)) =
        (prefixes.values.map componentCount).sum := by
    change
      (prefixes.values.flatMap (fun outer =>
        inner.values.map (fun seed => (outer, seed)))).countP
          (fun seed => event (outcome seed.1 seed.2)) =
        (prefixes.values.map componentCount).sum
    rw [List.countP_flatMap]
    apply congrArg List.sum
    apply List.map_congr_left
    intro outer _
    dsimp only [Function.comp_apply, componentCount]
    simpa only [Function.comp_apply] using
      (List.countP_map
        (l := inner.values)
        (f := fun seed => (outer, seed))
        (p := fun seed => event (outcome seed.1 seed.2)))
  have productCountCast :
      ((prefixes.product inner).values.countP
        (fun seed => event (outcome seed.1 seed.2)) : Rat) =
        (prefixes.values.map
          (fun outer => (componentCount outer : Rat))).sum := by
    rw [productCount, natCast_sum]
    simp only [List.map_map, Function.comp_def]
  change
    (prefixes.values.map (fun outer =>
      (componentCount outer : Rat) /
        (inner.cardinality : Rat))).sum /
        (prefixes.cardinality : Rat) =
      ((prefixes.product inner).values.countP
        (fun seed => event (outcome seed.1 seed.2)) : Rat) /
          ((prefixes.product inner).cardinality : Rat)
  rw [sum_map_div, productCountCast, Support.product_cardinality,
    Rat.natCast_mul]
  simp only [Rat.div_def]
  rw [Rat.inv_mul_rev]
  exact Rat.mul_assoc _ _ _

/-- Propositional-event form of
`sharedSupport_probabilityBool_eq_product`. -/
theorem Mixture.sharedSupport_probability_eq_product
    (prefixes : Support Prefix)
    (inner : Support Inner)
    (outcome : Prefix -> Inner -> Outcome)
    (event : Outcome -> Prop) :
    ({ prefixes := prefixes
       component := fun outer =>
        { Seed := Inner
          support := inner
          outcome := outcome outer } } :
      Mixture Prefix Outcome).probability event =
    ({ Seed := Prefix × Inner
       support := prefixes.product inner
       outcome := fun seed => outcome seed.1 seed.2 } :
      Experiment Outcome).probability event := by
  classical
  let mixture : Mixture Prefix Outcome :=
    { prefixes := prefixes
      component := fun outer =>
        { Seed := Inner
          support := inner
          outcome := outcome outer } }
  let productExperiment : Experiment Outcome :=
    { Seed := Prefix × Inner
      support := prefixes.product inner
      outcome := fun seed => outcome seed.1 seed.2 }
  let test : Outcome -> Bool := fun value =>
    if event value then true else false
  have eventEq : event = (fun value => test value = true) := by
    funext value
    apply propext
    simp [test]
  change mixture.probability event = productExperiment.probability event
  calc
    mixture.probability event =
        mixture.probability (fun value => test value = true) := by
          rw [eventEq]
    _ = mixture.probabilityBool test :=
      mixture.probability_bool_event test
    _ = productExperiment.probabilityBool test := by
      simpa only [mixture, productExperiment] using
        Mixture.sharedSupport_probabilityBool_eq_product
          prefixes inner outcome test
    _ = productExperiment.probability
          (fun value => test value = true) :=
      (productExperiment.probability_bool_event test).symm
    _ = productExperiment.probability event := by
      rw [eventEq]

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
