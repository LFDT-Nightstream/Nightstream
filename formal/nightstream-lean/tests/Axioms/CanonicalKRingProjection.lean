import Nightstream.Implementation.R1CS.Canonical.KRingProjection
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKRingProjection

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRingProjection.projectionRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KRingProjection.projectionRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KRingProjection.modulusCoefficients_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KRingProjection.modulusCoefficients_length

end NightstreamTests.Axioms.CanonicalKRingProjection
