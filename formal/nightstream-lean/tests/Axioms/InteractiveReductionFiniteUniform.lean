import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for finite-uniform rational probability
and exact expected-cost accounting.
-/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.probability_mono' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.probability_mono

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Mixture.loss_le_of_components' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Mixture.loss_le_of_components

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.expectedCost_le_iff_expectedCostAtMost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.expectedCost_le_iff_expectedCostAtMost
