import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection

/-!
Finite first-success extraction accounting for SuperNeo Appendix D.4.

Owns: inserting the guaranteed first-success predicate into a conditioned-pair
event, and the generic extraction inequality obtained from a raw iid mismatch
budget, a positive first-success floor, a fixed-first bad-event budget, and a
pointwise extraction cover.

Does not own: a protocol, a witness relation, coordinate forking, an infinite
or geometric rejection sampler, asymptotic running time, Fiat--Shamir, Rust,
R1CS, or constraints.

The conclusion is about the actual first-conditioned/fresh-second mixture.
Neither conditioned mismatch probability nor extraction probability is a
premise.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uOutcome

variable {Outcome : Type uOutcome}

/-- In the first-conditioned mixture, conjoining the first-success event does
not change an event that depends on the fresh second execution. -/
theorem Experiment.firstConditionedFreshSecond_first_success_and_second_marginal
    (experiment : Experiment Outcome)
    (success secondEvent : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ []) :
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
        (fun sample => success sample.1 && secondEvent sample.2) =
      experiment.probabilityBool secondEvent := by
  have componentEquality : forall firstSeed,
      firstSeed ∈ experiment.support.values.filter
        (fun seed => success (experiment.outcome seed)) ->
      experiment.probabilityBool (fun second =>
          success (experiment.outcome firstSeed) && secondEvent second) =
        experiment.probabilityBool secondEvent := by
    intro firstSeed member
    have firstAccepted : success (experiment.outcome firstSeed) = true :=
      (List.mem_filter.mp member).2
    have eventEquality :
        (fun second =>
          success (experiment.outcome firstSeed) && secondEvent second) =
        secondEvent := by
      funext second
      simp [firstAccepted]
    rw [eventEquality]
  calc
    (experiment.firstConditionedFreshSecond success nonempty).probabilityBool
          (fun sample => success sample.1 && secondEvent sample.2) =
        Mixture.probabilityBool
          (Experiment.firstConditionedFreshSecond experiment success nonempty)
          (fun sample => secondEvent sample.2) := by
            unfold Mixture.probabilityBool
              Experiment.firstConditionedFreshSecond
            congr 1
            apply congrArg List.sum
            apply List.map_congr_left
            intro firstSeed member
            exact componentEquality firstSeed member
    _ = experiment.probabilityBool secondEvent :=
      experiment.firstConditionedFreshSecond_second_marginal
        success nonempty secondEvent

/-- Generic finite Appendix-D.4 extraction inequality.

The raw iid mismatch budget is divided by the positive first-success floor.
The bad-event budget is proved for each fixed successful first seed and then
averaged by the conditioned mixture.  The only semantic premise is the
pointwise cover: two successful executions with no mismatch expose either the
declared extraction event or the declared bad event. -/
theorem extract_after_first_success
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (mismatch bad extracted : Outcome × Outcome -> Bool)
    (rawMismatchBudget badBudget successFloor : Rat)
    (successFloorPos : 0 < successFloor)
    (successFloorBound : successFloor <=
      experiment.probabilityBool success)
    (mismatchImpliesFirstSuccess : forall first second,
      mismatch (first, second) = true -> success first = true)
    (rawMismatchBound : experiment.iidPair.probabilityBool mismatch <=
      rawMismatchBudget)
    (fixedFirstBadBound : forall firstSeed,
      firstSeed ∈ experiment.support.values.filter
        (fun seed => success (experiment.outcome seed)) ->
      experiment.probabilityBool (fun second =>
        bad (experiment.outcome firstSeed, second)) <= badBudget)
    (cover : forall first second,
      success first = true -> success second = true ->
      mismatch (first, second) = false ->
      extracted (first, second) = true \/ bad (first, second) = true) :
    experiment.probabilityBool success -
        (badBudget + rawMismatchBudget / successFloor) <=
      Mixture.probabilityBool
        (Experiment.firstConditionedFreshSecond experiment success nonempty)
        extracted := by
  let conditioned := experiment.firstConditionedFreshSecond success nonempty
  have conditionedMismatchBound :
      conditioned.probabilityBool mismatch <=
        rawMismatchBudget / successFloor := by
    exact Experiment.firstConditionedFreshSecond_probabilityBool_le_div_floor
      experiment success nonempty mismatch mismatchImpliesFirstSuccess
      rawMismatchBudget successFloor rawMismatchBound
      successFloorPos successFloorBound
  have conditionedBadBound :
      conditioned.probabilityBool bad <= badBudget := by
    exact
      Experiment.firstConditionedFreshSecond_probabilityBool_le_of_fixedFirst
        experiment success nonempty bad badBudget fixedFirstBadBound
  have conditionedCombinedBadBound :
      conditioned.probabilityBool (fun sample =>
          bad sample || mismatch sample) <=
        badBudget + rawMismatchBudget / successFloor := by
    have exactUnionBound := conditioned.probabilityBool_or_le bad mismatch
    have replaceBad :
        conditioned.probabilityBool bad +
            conditioned.probabilityBool mismatch <=
          badBudget + conditioned.probabilityBool mismatch :=
      (Rat.add_le_add_right
        (c := conditioned.probabilityBool mismatch)).mpr conditionedBadBound
    have replaceMismatch :
        badBudget + conditioned.probabilityBool mismatch <=
          badBudget + rawMismatchBudget / successFloor :=
      (Rat.add_le_add_left (c := badBudget)).mpr conditionedMismatchBound
    exact Rat.le_trans exactUnionBound
      (Rat.le_trans replaceBad replaceMismatch)
  have pointwiseCover : forall sample,
      (success sample.1 && success sample.2) = true ->
        extracted sample = true \/
          (bad sample || mismatch sample) = true := by
    intro sample bothSuccessful
    have successFacts := Bool.and_eq_true_iff.mp bothSuccessful
    cases mismatchHolds : mismatch sample with
    | false =>
        rcases cover sample.1 sample.2 successFacts.1 successFacts.2
            mismatchHolds with extractedHolds | badHolds
        · exact Or.inl extractedHolds
        · exact Or.inr (by simp [badHolds])
    | true =>
        exact Or.inr (by simp)
  have extractionBound := conditioned.probabilityBool_sub_le_of_cover
    (fun sample => success sample.1 && success sample.2)
    extracted (fun sample => bad sample || mismatch sample)
    (badBudget + rawMismatchBudget / successFloor)
    pointwiseCover conditionedCombinedBadBound
  change
    conditioned.probabilityBool
          (fun sample => success sample.1 && success sample.2) -
        (badBudget + rawMismatchBudget / successFloor) <=
      conditioned.probabilityBool extracted at extractionBound
  rw [Experiment.firstConditionedFreshSecond_first_success_and_second_marginal
    experiment success success nonempty] at extractionBound
  exact extractionBound

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
