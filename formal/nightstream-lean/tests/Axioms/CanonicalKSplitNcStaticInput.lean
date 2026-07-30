import Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcStaticInput

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput.withDynamicClaims_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcStaticInput.withDynamicClaims_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput.rows_retarget' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcStaticInput.rows_retarget

end NightstreamTests.Axioms.CanonicalKSplitNcStaticInput
