import Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKProjectionTrace

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KProjectionTrace.Trace.exact_congr_below' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KProjectionTrace.Trace.exact_congr_below

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KProjectionTrace.PairColumns.productPolynomial_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KProjectionTrace.PairColumns.productPolynomial_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KProjectionTrace.Trace.identity_wellFormed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KProjectionTrace.Trace.identity_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KProjectionTrace.Trace.identity_ofLegacy' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KProjectionTrace.Trace.identity_ofLegacy

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KProjectionTrace.Trace.valid_ofLegacy' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KProjectionTrace.Trace.valid_ofLegacy

end NightstreamTests.Axioms.CanonicalKProjectionTrace
