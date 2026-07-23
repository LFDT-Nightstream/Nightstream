import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Focused executable and interface regression for finite-uniform rational
probability, pushforwards, mixtures, and exact expected costs.
-/

namespace tests.InteractiveReductionFiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

#check Support.cardinality_pos
#check Support.uniform
#check Experiment.map
#check Experiment.map_probabilityBool
#check Experiment.probability_bool_event
#check Experiment.toProbabilityExperiment
#check Mixture.map
#check Mixture.loss_le_of_components
#check Mixture.toProbabilityExperiment
#check Experiment.expectedCost_le_iff_totalCost_le
#check Experiment.expectedCost_le_iff_expectedCostAtMost
#check Experiment.ExpectedQueriesAtMost
#check le_div_iff_of_pos
#check div_le_iff_of_pos

def threeSeeds : Support Nat where
  values := [0, 1, 2]
  nodup := by decide
  nonempty := by decide

def uniformThree : Experiment Nat :=
  threeSeeds.uniform

def parity : Nat -> Nat :=
  fun value => value % 2

def parityExperiment : Experiment Nat :=
  uniformThree.map parity

example : threeSeeds.cardinality = 3 := by rfl

example : uniformThree.countBool (fun value => value % 2 == 0) = 2 := by
  rfl

example : uniformThree.probabilityBool (fun value => value % 2 == 0) =
    ratio 2 3 := by
  rfl

/-- The many-to-one parity map retains both even source seeds. -/
example : parityExperiment.probabilityBool (fun value => value == 0) =
    ratio 2 3 := by
  rfl

def twoSeeds : Support Nat where
  values := [0, 1]
  nodup := by decide
  nonempty := by decide

def outerSeeds : Support Bool where
  values := [false, true]
  nodup := by decide
  nonempty := by decide

def shiftedComponent (outer : Bool) : Experiment Nat :=
  match outer with
  | false => twoSeeds.uniform
  | true => twoSeeds.uniform.map (fun value => value + 1)

def shiftedMixture : Mixture Bool Nat where
  prefixes := outerSeeds
  component := shiftedComponent

/-- The two outer components contribute `1/2` and `0`; their uniform average
is kept structural here rather than asking the regression to normalize Rat. -/
example : shiftedMixture.probabilityBool (fun value => value == 0) =
    (ratio 1 2 + ratio 0 2) / 2 := by
  simp [shiftedMixture, Mixture.probabilityBool, shiftedComponent,
    outerSeeds, twoSeeds, Support.uniform, Experiment.map,
    Experiment.probabilityBool, Experiment.countBool,
    Support.cardinality, ratio]
  rw [Rat.add_zero]

def seedCost (seed : Nat) : Nat := seed

example : uniformThree.totalCost seedCost = 3 := by rfl

example : uniformThree.expectedCost seedCost = ratio 3 3 := by rfl

example : uniformThree.ExpectedCostAtMost seedCost 1 := by
  change 3 ≤ 1 * 3
  decide

def queryTrace (seed : Nat) : List Unit :=
  List.replicate seed ()

example : uniformThree.ExpectedQueriesAtMost queryTrace 1 := by
  change 3 ≤ 1 * 3
  decide

end tests.InteractiveReductionFiniteUniform
