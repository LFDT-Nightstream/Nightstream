import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.ProfileScope
import tests.Axioms.Support

/-! Fail-closed dependency checks for diagnostic/fixed-point separation. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_eq_diagnostic' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_eq_diagnostic

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactFieldCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactFieldCount_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_ne_steadyFixedPoint' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_ne_steadyFixedPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_ne_selectiveShape' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope.artifactProfile_ne_selectiveShape
