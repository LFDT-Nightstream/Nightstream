import Nightstream.Implementation.R1CS.Canonical.KBridge
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKBridge

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBridge.toPair_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KBridge.toPair_add

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBridge.toPair_mul' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KBridge.toPair_mul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBridge.toPair_eval' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KBridge.toPair_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KBridge.toPair_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KBridge.toPair_injective

end NightstreamTests.Axioms.CanonicalKBridge
