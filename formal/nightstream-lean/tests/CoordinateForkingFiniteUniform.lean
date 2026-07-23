import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment.Headline

/-!
Focused interface regression for the finite-uniform coordinate-forking theorem.

This deliberately checks theorem signatures only.  The underlying finite
probability and query-count arguments are exercised by their leaf regressions;
this file protects the public composition boundary without asking Lean to
evaluate a concrete transcript.
-/

set_option autoImplicit false

namespace tests.CoordinateForkingFiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform

#check base_probability_eq_uniform
#check bad_probability_le_sharp
#check expected_queries_at_most
#check finite_uniform_coordinate_forking

end tests.CoordinateForkingFiniteUniform
