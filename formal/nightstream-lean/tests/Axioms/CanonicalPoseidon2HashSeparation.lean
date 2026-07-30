import Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the rows that apply the prior/next separator.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2HashSeparation

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.priorColumn_ne_nextColumn' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.priorColumn_ne_nextColumn

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationRows_applies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationRows_applies

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.changed_tail_rejected' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.changed_tail_rejected

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationCost_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separation_applied_and_preserved' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separation_applied_and_preserved


/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.separationRows_eq_map_owners' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.separationRows_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.owners_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.owners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashSeparation.ownership_is_positional

end NightstreamTests.Axioms.CanonicalPoseidon2HashSeparation
