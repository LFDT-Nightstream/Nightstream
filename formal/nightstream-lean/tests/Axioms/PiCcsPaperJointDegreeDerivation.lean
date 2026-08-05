import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Types
import tests.Axioms.Support

/-!
Fail-closed audit for the two joint-polynomial degree facts that carry
`FOLD-PICCS-JOINT`'s expected-syntactic-degree obligation and were previously
unguarded.

Owns: the exact gamma-degree of the signed coefficient polynomial, and the
congruence showing the SumCheck degree ceiling is *derived* from the
constraint polynomial rather than supplied by a caller.

Does not own: round representability at that ceiling
(`tests/Axioms/PiCcsPaperJointDegreeWidth.lean`), root counting, or any
production degree claim.
-/

namespace NightstreamTests.Axioms.PiCcsPaperJointDegreeDerivation

open NightstreamTests.Axioms

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial.polynomial_degreeUpperBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial.polynomial_degreeUpperBound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_eq_of_terms_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_eq_of_terms_eq

end NightstreamTests.Axioms.PiCcsPaperJointDegreeDerivation
