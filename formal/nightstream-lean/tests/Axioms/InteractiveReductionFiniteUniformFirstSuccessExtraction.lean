import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessExtraction
import tests.Axioms.Support

/-!
Trusted-dependency probes for finite first-success extraction.

The expected dependency sets were recorded from a focused coordinated build.
Any later drift fails closed.
-/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.firstConditionedFreshSecond_first_success_and_second_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.firstConditionedFreshSecond_first_success_and_second_marginal

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.extract_after_first_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.extract_after_first_success
