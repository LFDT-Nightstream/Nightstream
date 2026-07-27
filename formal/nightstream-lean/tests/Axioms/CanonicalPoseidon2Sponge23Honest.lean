import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Honest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest.honest_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge23Honest.honest_satisfies

end NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Honest
