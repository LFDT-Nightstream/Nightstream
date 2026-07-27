import Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance
import tests.Axioms.Support

/-!
Guards for the Rust-conformance check.

These are deliberately the *only* guards that reach the generated artifact. A
canonical theorem that started depending on the artifact would show up as a new
import edge, not as a change here.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2RustConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance.rust_matches_lean_initial' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2RustConformance.rust_matches_lean_initial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance.rust_matches_lean_internal' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2RustConformance.rust_matches_lean_internal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance.rust_matches_lean_terminal' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2RustConformance.rust_matches_lean_terminal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance.rust_matches_lean_diagonal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2RustConformance.rust_matches_lean_diagonal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RustConformance.rust_diagonal_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2RustConformance.rust_diagonal_length

end NightstreamTests.Axioms.CanonicalPoseidon2RustConformance
