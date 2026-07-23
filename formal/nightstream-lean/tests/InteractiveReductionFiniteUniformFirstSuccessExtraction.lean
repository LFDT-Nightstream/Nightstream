import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessExtraction

/-!
Focused interface and algebra regression for finite first-success extraction.
The fixture keeps every probability budget expressed through the experiment
itself, so the test exercises theorem composition without asking the kernel to
normalize rational constants.
-/

set_option autoImplicit false

namespace tests.InteractiveReductionFiniteUniformFirstSuccessExtraction

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

#check Experiment.firstConditionedFreshSecond_first_success_and_second_marginal
#check extract_after_first_success

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

def mismatch (_ : Nat × Nat) : Bool :=
  false

def bad (_ : Nat × Nat) : Bool :=
  false

def extracted (sample : Nat × Nat) : Bool :=
  succeeds sample.1 && succeeds sample.2

example :
    (baseExperiment.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool
          (fun sample => succeeds sample.1 && succeeds sample.2) =
      baseExperiment.probabilityBool succeeds := by
  exact
    Experiment.firstConditionedFreshSecond_first_success_and_second_marginal
      baseExperiment succeeds succeeds successfulSeedsNonempty

example :
    baseExperiment.probabilityBool succeeds -
        (baseExperiment.probabilityBool (fun _ => false) +
          baseExperiment.iidPair.probabilityBool mismatch /
            baseExperiment.probabilityBool succeeds) ≤
      (baseExperiment.firstConditionedFreshSecond succeeds
        successfulSeedsNonempty).probabilityBool extracted := by
  apply extract_after_first_success
    baseExperiment succeeds successfulSeedsNonempty mismatch bad extracted
    (baseExperiment.iidPair.probabilityBool mismatch)
    (baseExperiment.probabilityBool (fun _ => false))
    (baseExperiment.probabilityBool succeeds)
  · exact Experiment.probabilityBool_pos_of_filter_nonempty
      baseExperiment succeeds successfulSeedsNonempty
  · exact Rat.le_refl
  · intro first second impossible
    change false = true at impossible
    exact Bool.noConfusion impossible
  · exact Rat.le_refl
  · intro _firstSeed _firstMember
    exact Rat.le_refl
  · intro first second firstSuccess secondSuccess _
    exact Or.inl (Bool.and_eq_true_iff.mpr ⟨firstSuccess, secondSuccess⟩)

end tests.InteractiveReductionFiniteUniformFirstSuccessExtraction
