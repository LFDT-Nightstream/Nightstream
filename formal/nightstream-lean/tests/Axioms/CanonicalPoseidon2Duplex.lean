import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the Fiat-Shamir duplex model.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Duplex

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.permute_absorbed' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.permute_absorbed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.guarded_absorbed_lt' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.guarded_absorbed_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.write_index_lt_rate' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.write_index_lt_rate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.cursor_le_rate' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.cursor_le_rate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.cursor_le_rate_list' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.cursor_le_rate_list

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.empty_cursor' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.empty_cursor

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.capacity_untouched' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.capacity_untouched

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.gate_absorbed' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.gate_absorbed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.challengeField_state' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.challengeField_state

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.challengeField_cursor' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.challengeField_cursor

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.duplex_absorb_is_overwrite' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Duplex.duplex_absorb_is_overwrite

end NightstreamTests.Axioms.CanonicalPoseidon2Duplex
