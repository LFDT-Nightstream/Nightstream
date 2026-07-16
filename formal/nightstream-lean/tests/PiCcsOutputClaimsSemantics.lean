import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics

/-!
Kernel-facing regression for the independent Phi81 `PiCCS` output-claim
semantics. These checks expose only the source-derived target; they do not
claim that verifier acceptance or production constraints establish it.
-/

namespace tests.PiCcsOutputClaimsSemantics

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims

#check yRingTableForMatrixSource
#check yRingForMatrixSource
#check freshYZcolTerm_tail_zero
#check canonicalClaims_yRingBoundToSources
#check canonicalClaims_yZcolBoundToSources
#check canonicalClaims_boundToSources
#check eq_canonicalClaims_of_boundToSources
#check boundToSources_iff_eq_canonicalClaims

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.freshYZcolTerm_tail_zero' depends on axioms: [propext] -/
#guard_msgs in
#print axioms freshYZcolTerm_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.boundToSources_iff_eq_canonicalClaims' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms boundToSources_iff_eq_canonicalClaims

end tests.PiCcsOutputClaimsSemantics
