import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcBlockLaneRows

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneRows.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows.rows_cost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneRows.rows_cost

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows.auxiliary_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneRows.auxiliary_count

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows.accepted_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcBlockLaneRows.accepted_of_rows

end NightstreamTests.Axioms.CanonicalKSplitNcBlockLaneRows
