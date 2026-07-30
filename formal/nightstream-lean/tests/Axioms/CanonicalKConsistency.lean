import Nightstream.Implementation.R1CS.Canonical.KConsistency
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKConsistency

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyRows_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyRows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyCost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyCost_rows


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConsistency.consistencyRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KConsistency.consistencyRows_conservation

end NightstreamTests.Axioms.CanonicalKConsistency
