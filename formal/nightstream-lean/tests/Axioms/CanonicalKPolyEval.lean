import Nightstream.Implementation.R1CS.Canonical.KPolyEval
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKPolyEval

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.polyEval_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KPolyEval.polyEval_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.polyEval_polyAdd' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyEval.polyEval_polyAdd

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.polyEval_polyScale' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyEval.polyEval_polyScale

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.polyEval_polyMul' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyEval.polyEval_polyMul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.hornerValue_eq_polyEval' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPolyEval.hornerValue_eq_polyEval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPolyEval.polyEval_quotientForm' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPolyEval.polyEval_quotientForm

end NightstreamTests.Axioms.CanonicalKPolyEval
