import Nightstream.Implementation.R1CS.Canonical.KMul
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKMul

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMul.rows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KMul.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMul.frame_products' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KMul.frame_products

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMul.outLow_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KMul.outLow_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMul.outHigh_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KMul.outHigh_sound

end NightstreamTests.Axioms.CanonicalKMul
