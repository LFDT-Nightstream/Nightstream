import Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness
import tests.Axioms.Support

/-!
Fail-closed axiom guard for Poseidon2 witness uniqueness.

Every report is measured: the expected text was produced by running
`#audit_axioms` and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Uniqueness

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.rowHolds_congr' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.rowHolds_congr

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.satisfies_congr' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.satisfies_congr

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.canonicalProgram_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.canonicalProgram_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.scheduleOf_eval' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.scheduleOf_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.sboxColumn_forced' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.sboxColumn_forced

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.canonicalProgram_exec_iff_spec' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.canonicalProgram_exec_iff_spec

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Uniqueness.permutationProgram_exec_iff_spec' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Uniqueness.permutationProgram_exec_iff_spec

end NightstreamTests.Axioms.CanonicalPoseidon2Uniqueness
