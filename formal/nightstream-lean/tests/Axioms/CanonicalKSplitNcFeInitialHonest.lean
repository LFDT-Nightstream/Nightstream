import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKSplitNcFeInitialHonest

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest.witness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeInitialHonest.witness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitialHonest.rows_honest_of_binding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeInitialHonest.rows_honest_of_binding

end NightstreamTests.Axioms.CanonicalKSplitNcFeInitialHonest
