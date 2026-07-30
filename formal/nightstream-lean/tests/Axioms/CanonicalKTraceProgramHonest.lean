import Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKTraceProgramHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest.traceRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgramHonest.traceRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest.batchWitness_preserves_below' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgramHonest.batchWitness_preserves_below

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest.rowsFrom_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceProgramHonest.rowsFrom_honest

end NightstreamTests.Axioms.CanonicalKTraceProgramHonest
