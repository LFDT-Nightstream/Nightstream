import Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Compact

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact.computes_reference' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Compact.computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact.of_canonical_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Compact.of_canonical_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact.honest_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Compact.honest_holds

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact.canonical_activeColumns_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Poseidon2Compact.canonical_activeColumns_exact

end NightstreamTests.Axioms.CanonicalPoseidon2Compact
