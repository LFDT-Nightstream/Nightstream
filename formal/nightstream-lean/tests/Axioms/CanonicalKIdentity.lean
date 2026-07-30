import Nightstream.Implementation.R1CS.Canonical.KIdentity
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKIdentity

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KIdentity.identityRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KIdentity.identityRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KIdentity.identityRows_length_of_degree' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KIdentity.identityRows_length_of_degree

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KIdentity.identityRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KIdentity.identityRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KIdentity.identityRows_is_projection_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KIdentity.identityRows_is_projection_eval

end NightstreamTests.Axioms.CanonicalKIdentity
