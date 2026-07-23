import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.TruncatedRejection

/-!
Focused executable and interface regression for finite truncated rejection
sampling.  The two-seed fixture has one successful seed; at cutoff three its
eight equiprobable tapes use fourteen verifier calls in total.
-/

set_option autoImplicit false

namespace tests.InteractiveReductionFiniteUniformTruncatedRejection

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

#check Support.cartesianPower_cardinality
#check Support.cartesianPower_tape_length
#check firstSuccessCalls_le_length
#check firstSuccessCalls_map
#check Experiment.truncatedQueryCost_eq_attemptsUsed
#check Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio
#check Experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess
#check Experiment.truncatedFirstSuccess_expectedQueries_le_inverseFloor

def twoSeeds : Support Nat where
  values := [0, 1]
  nodup := by decide
  nonempty := by decide

def baseExperiment : Experiment Nat :=
  twoSeeds.uniform

def succeeds (value : Nat) : Bool :=
  value == 0

def successfulSeedsNonempty :
    baseExperiment.support.values.filter
      (fun seed => succeeds (baseExperiment.outcome seed)) ≠ [] := by
  decide

example :
    (twoSeeds.cartesianPower 3).cardinality = 2 ^ 3 := by
  exact Support.cartesianPower_cardinality twoSeeds 3

example :
    (baseExperiment.truncatedFirstSuccess succeeds 3).totalCost
      (baseExperiment.truncatedQueryCost succeeds 3) = 14 := by
  decide

example (attemptLimit : Nat) :
    (baseExperiment.truncatedFirstSuccess succeeds attemptLimit).expectedCost
        (baseExperiment.truncatedQueryCost succeeds attemptLimit) ≤
      ratio baseExperiment.support.cardinality
        (baseExperiment.countBool succeeds) := by
  exact Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio
    baseExperiment succeeds successfulSeedsNonempty attemptLimit

example (attemptLimit : Nat) :
    (baseExperiment.truncatedFirstSuccess succeeds attemptLimit).expectedCost
        (baseExperiment.truncatedQueryCost succeeds attemptLimit) ≤
      1 / baseExperiment.probabilityBool succeeds := by
  exact Experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess
    baseExperiment succeeds successfulSeedsNonempty attemptLimit

end tests.InteractiveReductionFiniteUniformTruncatedRejection
