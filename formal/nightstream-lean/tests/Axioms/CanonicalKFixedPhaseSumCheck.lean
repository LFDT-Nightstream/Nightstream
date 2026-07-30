import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheck

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck.chainRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheck.chainRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck.chainRows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheck.chainRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck.chainCost_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheck.chainCost_rows

end NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheck
