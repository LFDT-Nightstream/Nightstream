import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientCertificate

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate.table_shapes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2PartialCoefficientCertificate.table_shapes

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate.selected_partial_coefficients_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  Poseidon2PartialCoefficientCertificate.selected_partial_coefficients_check

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate.tableAt_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2PartialCoefficientCertificate.tableAt_nonzero

end NightstreamTests.Axioms.CanonicalPoseidon2PartialCoefficientCertificate
