import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcOperationalRows

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows.accepted_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcOperationalRows.accepted_of_rows

end NightstreamTests.Axioms.CanonicalKSplitNcOperationalRows
