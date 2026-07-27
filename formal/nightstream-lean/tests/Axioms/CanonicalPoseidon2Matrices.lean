import Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the concrete Poseidon2 linear layers.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Matrices

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.mat4_nonzero' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.mat4_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.externalMatrix_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.externalMatrix_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.externalMatrix_lt' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.externalMatrix_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.internalDiag_half_inverse' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.internalDiag_half_inverse

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.internalDiag_neg_half' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.internalDiag_neg_half

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.internalMatrix_nonzero' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.internalMatrix_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices.internalMatrix_lt' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Matrices.internalMatrix_lt

end NightstreamTests.Axioms.CanonicalPoseidon2Matrices
