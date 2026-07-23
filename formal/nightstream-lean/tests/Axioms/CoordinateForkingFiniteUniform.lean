import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment.Headline
import tests.Axioms.Support

/-!
Trusted-dependency probe for the finite-uniform coordinate-forking headline.

The expected dependency set is fail-closed: any later addition or removal must
be reviewed explicitly rather than silently changing the theorem's trust base.
-/

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.finite_uniform_coordinate_forking' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.finite_uniform_coordinate_forking
