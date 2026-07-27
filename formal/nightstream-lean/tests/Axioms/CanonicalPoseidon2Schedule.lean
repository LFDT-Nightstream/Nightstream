import Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the concrete Poseidon2 round schedule
and the permutation program built from it.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Schedule

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.sboxIndex_partition' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.sboxIndex_partition

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.initialSboxIndex_roundtrip' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.initialSboxIndex_roundtrip

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.partialSboxIndex_roundtrip' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.partialSboxIndex_roundtrip

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.terminalSboxIndex_roundtrip' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.terminalSboxIndex_roundtrip

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.canonicalProgram_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.canonicalProgram_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.canonicalProgram_cost' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.canonicalProgram_cost

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.canonicalProgram_sbox_chains' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.canonicalProgram_sbox_chains

/-! Generalized entry state for the sponge. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.initialState_eq_from' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.initialState_eq_from

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.initialStateFrom_succ_entry_irrelevant' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.initialStateFrom_succ_entry_irrelevant

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.initialStateFrom_halfFull_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.initialStateFrom_halfFull_eq

/-! Schedule and program on a carried entry. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.scheduleOfFrom_port_entry' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.scheduleOfFrom_port_entry

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule.canonicalProgramFrom_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Schedule.canonicalProgramFrom_length

end NightstreamTests.Axioms.CanonicalPoseidon2Schedule
