import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.TruncatedRejection
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency probes for finite truncated rejection sampling.
The expected sets were recorded from a focused build.
-/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.cartesianPower_cardinality' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.cartesianPower_cardinality

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedQueryCost_eq_attemptsUsed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedQueryCost_eq_attemptsUsed

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_inverseFloor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Experiment.truncatedFirstSuccess_expectedQueries_le_inverseFloor
