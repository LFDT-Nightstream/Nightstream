import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Sponge23

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout_wellFormed' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23.layout_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23.program_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program_computes_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23.program_computes_digest

end NightstreamTests.Axioms.CanonicalPoseidon2Sponge23
