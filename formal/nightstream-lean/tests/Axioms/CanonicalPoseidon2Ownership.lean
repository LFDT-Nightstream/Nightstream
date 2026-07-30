import Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership
import tests.Axioms.Support

/-!
Fail-closed dependency gate for positional receipt ownership.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Ownership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.ownership_is_positional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.ownership_is_positional

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.normalizedCanonicalProgram_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.normalizedCanonicalProgram_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.canonicalProgram_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.canonicalProgram_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.allOwners_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.allOwners_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.allOwners_index_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.allOwners_index_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.allOwners_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.allOwners_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.allOwners_split' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.allOwners_split

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.nodup_of_map_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.nodup_of_map_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.sboxRows_eq_map' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.sboxRows_eq_map

/-! Carried-entry and sponge ownership. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.ownership_is_positional_from' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.ownership_is_positional_from

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.canonicalProgramFrom_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.canonicalProgramFrom_eq_map_owners


/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership.nodup_map_of_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Ownership.nodup_map_of_injective

end NightstreamTests.Axioms.CanonicalPoseidon2Ownership
