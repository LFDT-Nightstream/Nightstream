import Nightstream.Implementation.R1CS.Canonical.KPolyHom
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKPolyHom

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyHom.polyEval_singleton' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KPolyHom.polyEval_singleton

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyHom.hornerValue_singleton' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyHom.hornerValue_singleton

end NightstreamTests.Axioms.CanonicalKPolyHom
