import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Product
import tests.Axioms.Support

/-!
Fail-closed dependency probes for exact finite Cartesian supports. The
expected sets were recorded from a focused build.
-/

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.mem_product_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Support.mem_product_iff

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.product_cardinality' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Support.product_cardinality

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.product_uniform_probabilityBool_first' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Support.product_uniform_probabilityBool_first

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Support.product_uniform_probabilityBool_second' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Support.product_uniform_probabilityBool_second
