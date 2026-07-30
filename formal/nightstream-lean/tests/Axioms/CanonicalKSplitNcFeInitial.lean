import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcFeInitial

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeInitial.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeInitial.rows_sound

end NightstreamTests.Axioms.CanonicalKSplitNcFeInitial
