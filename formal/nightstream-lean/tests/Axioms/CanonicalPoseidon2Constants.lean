import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Constants

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.initial_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2CanonicalConstants.initial_shape

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.internal_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2CanonicalConstants.internal_shape

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.terminal_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2CanonicalConstants.terminal_shape

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected_canonical' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2CanonicalConstants.selected_canonical

end NightstreamTests.Axioms.CanonicalPoseidon2Constants
