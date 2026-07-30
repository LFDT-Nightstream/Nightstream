import Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the assembled canonical Poseidon2 permutation
program.  No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Program

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.sboxColumn_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.sboxColumn_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.auxiliaryColumns_length_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Program.auxiliaryColumns_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.permutationProgram_length_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Program.permutationProgram_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.everyPermutationRow_has_owner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.everyPermutationRow_has_owner

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.everyPermutationColumn_has_exact_owner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.everyPermutationColumn_has_exact_owner

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.permutationProgram_sbox_chains' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.permutationProgram_sbox_chains

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.permutationProgram_cost_eq_receiptFold' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Program.permutationProgram_cost_eq_receiptFold


/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.sboxProgram_writes_sboxColumn' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.sboxProgram_writes_sboxColumn

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.sboxProgram_writes_auxiliaryColumns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.sboxProgram_writes_auxiliaryColumns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Program.permutationProgram_writes_auxiliaryColumns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Program.permutationProgram_writes_auxiliaryColumns

end NightstreamTests.Axioms.CanonicalPoseidon2Program
