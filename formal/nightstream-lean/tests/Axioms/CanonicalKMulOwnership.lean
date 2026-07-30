import Nightstream.Implementation.R1CS.Canonical.KMulOwnership
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKMulOwnership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulOwnership.rows_eq_map_owners' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KMulOwnership.rows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulOwnership.allOwners_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KMulOwnership.allOwners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulOwnership.rows_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KMulOwnership.rows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulOwnership.ownedRow_target' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KMulOwnership.ownedRow_target

end NightstreamTests.Axioms.CanonicalKMulOwnership
