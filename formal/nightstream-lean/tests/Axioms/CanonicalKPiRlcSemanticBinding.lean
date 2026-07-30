import Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKPiRlcSemanticBinding

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding.exact_output_eq_phi81Combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcSemanticBinding.exact_output_eq_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding.equations_of_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcSemanticBinding.equations_of_exact

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding.equations_or_badRoot_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcSemanticBinding.equations_or_badRoot_of_rows

end NightstreamTests.Axioms.CanonicalKPiRlcSemanticBinding
