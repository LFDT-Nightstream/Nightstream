import Nightstream.Implementation.R1CS.Canonical.KEquality
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKEquality

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KEquality.rows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KEquality.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KEquality.rows_eq_map_owners' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KEquality.rows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KEquality.rows_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KEquality.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KEquality.rows_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KEquality.rows_complete

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KEquality.rows_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KEquality.rows_conservation

end NightstreamTests.Axioms.CanonicalKEquality
