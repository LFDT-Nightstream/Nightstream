import Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation
import tests.Axioms.Support

/-!
Fail-closed dependency gate for whole-program column conservation.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2ProgramConservation

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation.normalizedCanonicalProgram_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ProgramConservation.normalizedCanonicalProgram_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation.ownedRow_operand_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ProgramConservation.ownedRow_operand_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation.singleton_mentions_lt' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2ProgramConservation.singleton_mentions_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ProgramConservation.constantWire_lt' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2ProgramConservation.constantWire_lt

end NightstreamTests.Axioms.CanonicalPoseidon2ProgramConservation
