import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKConcreteBridge

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteBridge.ofConcrete_add

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_mul' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteBridge.ofConcrete_mul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_agrees_with_toPair' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteBridge.ofConcrete_agrees_with_toPair

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_sub' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KConcreteBridge.ofConcrete_sub

end NightstreamTests.Axioms.CanonicalKConcreteBridge
