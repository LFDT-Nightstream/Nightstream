import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKConcreteFixedPhaseBridge

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge.toProjection_mul' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteFixedPhaseBridge.toProjection_mul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge.evaluate_mapPolynomial' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteFixedPhaseBridge.evaluate_mapPolynomial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge.chain_toProjection' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteFixedPhaseBridge.chain_toProjection

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge.chain_iff_toProjection' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteFixedPhaseBridge.chain_iff_toProjection

end NightstreamTests.Axioms.CanonicalKConcreteFixedPhaseBridge
