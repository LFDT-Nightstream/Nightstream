import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperational
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcOperational

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcOperational.accepted_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcOperational.accepted_of_rows

end NightstreamTests.Axioms.CanonicalKSplitNcOperational
