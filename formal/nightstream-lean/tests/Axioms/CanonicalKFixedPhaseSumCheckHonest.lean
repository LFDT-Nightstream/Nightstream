import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheckHonest

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest.hornerCarried_below_next' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheckHonest.hornerCarried_below_next

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest.chainWitness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheckHonest.chainWitness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest.chainWitness_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFixedPhaseSumCheckHonest.chainWitness_satisfies

end NightstreamTests.Axioms.CanonicalKFixedPhaseSumCheckHonest
