import Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2ExactCoefficients

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients.selected_constants_nonzero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2ExactCoefficients.selected_constants_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients.selected_first_round_lengths' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ExactCoefficients.selected_first_round_lengths

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients.fieldScheduledSizes_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ExactCoefficients.fieldScheduledSizes_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients.program_nonzero_coefficient_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2ExactCoefficients.program_nonzero_coefficient_count

end NightstreamTests.Axioms.CanonicalPoseidon2ExactCoefficients
