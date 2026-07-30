import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheckSupport

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport.chainRows_columns_below_end' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheckSupport.chainRows_columns_below_end

end NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheckSupport
