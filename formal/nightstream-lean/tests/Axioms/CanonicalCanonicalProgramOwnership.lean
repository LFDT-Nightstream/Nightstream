import Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the assembly's positional row ownership.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.

`owners_nodup` and `ownership_is_positional` carry `Classical.choice`, which is
parity with `Poseidon2Ownership.ownership_is_positional` and
`PiDecOwnership.ownership_is_positional` rather than a widening.
-/

namespace NightstreamTests.Axioms.CanonicalCanonicalProgramOwnership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.rows_eq_parts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.rows_eq_parts

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.rows_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.rows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.owners_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.owners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.duplicating_selection_has_distinct_receipts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.duplicating_selection_has_distinct_receipts

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership.parts_have_distinct_receipts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgramOwnership.parts_have_distinct_receipts

end NightstreamTests.Axioms.CanonicalCanonicalProgramOwnership
