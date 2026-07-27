import Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2HonestFrom

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom.honest_satisfies_normalizedFrom' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HonestFrom.honest_satisfies_normalizedFrom

end NightstreamTests.Axioms.CanonicalPoseidon2HonestFrom
