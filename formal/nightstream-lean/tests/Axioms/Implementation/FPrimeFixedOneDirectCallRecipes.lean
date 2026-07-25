import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the selected direct fixed-one call
recipes and their explicit incomplete-call boundary.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.certifiedSubset' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.certifiedSubset

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.allRecipes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.allRecipes

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.remainingCalls_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.remainingCalls_exact
