import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Ownership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.program_eq_positional_receipts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23Ownership.program_eq_positional_receipts

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumns_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23Ownership.temporaryColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.inputs_disjoint_temporaries' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23Ownership.inputs_disjoint_temporaries

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.program_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23Ownership.program_conservation

end NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Ownership
