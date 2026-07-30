import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminal
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcFeTerminal

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminal.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminal.rows_sound

end NightstreamTests.Axioms.CanonicalKSplitNcFeTerminal
