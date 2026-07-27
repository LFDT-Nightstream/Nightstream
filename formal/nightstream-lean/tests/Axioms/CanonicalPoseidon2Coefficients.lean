import Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the Poseidon2 coefficient count.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Coefficients

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.canonicalProgram_termCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.canonicalProgram_termCount

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.sboxProgram_termCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.sboxProgram_termCount

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.bindingProgram_termCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.bindingProgram_termCount

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.sboxRows_termCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.sboxRows_termCount

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.bindRow_termCount' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.bindRow_termCount

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.programTermCount_append' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.programTermCount_append

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.programTermCount_flatMap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.programTermCount_flatMap

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.flatten_map_singleton' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.flatten_map_singleton

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.normalize_length_applyMatrix_singletons' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.normalize_length_applyMatrix_singletons

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.terminalOutput_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.terminalOutput_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.finalState_normalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.finalState_normalize_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.finalSizes_sum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.finalSizes_sum

/-! Structural no-cancellation for full-round states. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.fieldNormalize_length_applyMatrix_singletons' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.fieldNormalize_length_applyMatrix_singletons

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients.finalState_fieldNormalize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Coefficients.finalState_fieldNormalize_length

end NightstreamTests.Axioms.CanonicalPoseidon2Coefficients
