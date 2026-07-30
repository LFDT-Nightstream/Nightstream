import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpoint
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcNcEndpoint

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpoint.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcEndpoint.rows_sound

end NightstreamTests.Axioms.CanonicalKSplitNcNcEndpoint
