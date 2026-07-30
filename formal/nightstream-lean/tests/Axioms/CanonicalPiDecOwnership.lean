import Nightstream.Implementation.R1CS.Canonical.PiDecOwnership
import tests.Axioms.Support

/-!
Fail-closed axiom guard for Π_DEC's positional row ownership.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.

`owners_nodup` and `ownership_is_positional` carry `Classical.choice`, which is
parity with the established analogue `Poseidon2Ownership.ownership_is_positional`
rather than a widening.
-/

namespace NightstreamTests.Axioms.CanonicalPiDecOwnership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.map_getD_range' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecOwnership.map_getD_range

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.rows_eq_atoms' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PiDecOwnership.rows_eq_atoms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.rows_eq_map_owners' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecOwnership.rows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.owners_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecOwnership.owners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecOwnership.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecOwnership.duplicate_values_have_distinct_receipts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PiDecOwnership.duplicate_values_have_distinct_receipts

end NightstreamTests.Axioms.CanonicalPiDecOwnership
