import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection

/-!
Focused executable and interface regression for finite Appendix-D.4 rejection
conditioning.  The fixture has four equally likely executions, of which two
satisfy the first-run success predicate.
-/

set_option autoImplicit false

namespace tests.InteractiveReductionFiniteUniformRejection

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

#check Support.filterBool
#check Experiment.conditionBool
#check Experiment.iidPair
#check Experiment.firstConditionedFreshSecond
#check Experiment.firstConditionedFreshSecond_probabilityBool_eq_div
#check Experiment.firstConditionedFreshSecond_probabilityBool_le_div_floor
#check Experiment.firstConditionedFreshSecond_probabilityBool_le_of_fixedFirst
#check Experiment.firstConditionedFreshSecond_first_success
#check Experiment.firstConditionedFreshSecond_second_marginal
#check Experiment.probabilityBool_sub_le_of_cover
#check Mixture.probabilityBool_sub_le_of_cover

def fourSeeds : Support Nat where
  values := [0, 1, 2, 3]
  nodup := by decide
  nonempty := by decide

def uniformFour : Experiment Nat :=
  fourSeeds.uniform

def succeeds (value : Nat) : Bool :=
  decide (value < 2)

def successfulSeedsNonempty :
    uniformFour.support.values.filter
      (fun seed => succeeds (uniformFour.outcome seed)) ≠ [] := by
  decide

def firstSucceeds (sample : Nat × Nat) : Bool :=
  succeeds sample.1

def secondIsZero (sample : Nat × Nat) : Bool :=
  sample.2 == 0

theorem uniformFour_success_probability :
    uniformFour.probabilityBool succeeds = ratio 2 4 := by
  rfl

theorem iid_first_success_probability :
    uniformFour.iidPair.probabilityBool firstSucceeds =
      uniformFour.probabilityBool succeeds := by
  exact Experiment.iidPair_first_marginal uniformFour succeeds

/-- The exact finite ratio is exposed through the public interface, not
reproved by fixture computation. -/
example :
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool firstSucceeds =
      uniformFour.iidPair.probabilityBool firstSucceeds /
        uniformFour.probabilityBool succeeds := by
  apply Experiment.firstConditionedFreshSecond_probabilityBool_eq_div
  intro first _ eventHolds
  exact eventHolds

/-- Filtering derives, rather than assumes, that the first run succeeds with
probability one. -/
example :
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool firstSucceeds = 1 := by
  exact Experiment.firstConditionedFreshSecond_first_success
    uniformFour succeeds successfulSeedsNonempty

/-- The fresh second execution retains the original one-in-four marginal. -/
example :
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool secondIsZero = ratio 1 4 := by
  calc
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool secondIsZero =
      uniformFour.probabilityBool (fun value => value == 0) :=
        Experiment.firstConditionedFreshSecond_second_marginal
          uniformFour succeeds successfulSeedsNonempty
            (fun value => value == 0)
    _ = ratio 1 4 := by rfl

example :
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool secondIsZero =
      uniformFour.probabilityBool (fun value => value == 0) := by
  exact Experiment.firstConditionedFreshSecond_second_marginal
    uniformFour succeeds successfulSeedsNonempty (fun value => value == 0)

/-- The raw first-success budget divided by the same positive success floor
produces the conditioned ratio bound required by Appendix D.4. -/
example :
    (uniformFour.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool firstSucceeds ≤
      uniformFour.probabilityBool succeeds /
        uniformFour.probabilityBool succeeds := by
  refine Experiment.firstConditionedFreshSecond_probabilityBool_le_div_floor
    uniformFour succeeds successfulSeedsNonempty firstSucceeds ?_
      (uniformFour.probabilityBool succeeds)
      (uniformFour.probabilityBool succeeds) ?_ ?_ ?_
  · intro first _ eventHolds
    exact eventHolds
  · rw [iid_first_success_probability]
    exact Rat.le_refl
  · exact Experiment.probabilityBool_pos_of_filter_nonempty
      uniformFour succeeds successfulSeedsNonempty
  · exact Rat.le_refl

end tests.InteractiveReductionFiniteUniformRejection
