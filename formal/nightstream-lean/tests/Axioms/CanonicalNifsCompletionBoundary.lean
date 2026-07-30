import Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalNifsCompletionBoundary

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary.acceptingVerifier_result' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsCompletionBoundary.acceptingVerifier_result

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary.rejectingVerifier_result' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsCompletionBoundary.rejectingVerifier_result

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary.setupVerifier_is_a_real_choice' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsCompletionBoundary.setupVerifier_is_a_real_choice

end NightstreamTests.Axioms.CanonicalNifsCompletionBoundary
