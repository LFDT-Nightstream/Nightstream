import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the emitted field-canonical program.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Normalized

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.rowHolds_normalizeRow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.rowHolds_normalizeRow

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.satisfies_normalizeProgram' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.satisfies_normalizeProgram

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.mentions_normalizeRow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.mentions_normalizeRow

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizeRow_entries' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizeRow_entries

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.rawTermCount_normalizeRow_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.rawTermCount_normalizeRow_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.rawProgramTermCount_normalizeProgram_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.rawProgramTermCount_normalizeProgram_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizedCanonicalProgram_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizedCanonicalProgram_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizedCanonicalProgramFrom_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizedCanonicalProgramFrom_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizedCanonicalProgram_termCount_le' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizedCanonicalProgram_termCount_le

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizedCanonicalProgram_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizedCanonicalProgram_computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.normalizedCanonicalProgramFrom_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.normalizedCanonicalProgramFrom_computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.honest_satisfies_normalized' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Normalized.honest_satisfies_normalized

end NightstreamTests.Axioms.CanonicalPoseidon2Normalized
