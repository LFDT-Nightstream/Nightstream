import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientBridge

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge.partialState_basis_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2PartialCoefficientBridge.partialState_basis_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge.partialState_normalized_coefficients_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Poseidon2PartialCoefficientBridge.partialState_normalized_coefficients_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge.partialState_fieldNormalize_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Poseidon2PartialCoefficientBridge.partialState_fieldNormalize_length

end NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientBridge
