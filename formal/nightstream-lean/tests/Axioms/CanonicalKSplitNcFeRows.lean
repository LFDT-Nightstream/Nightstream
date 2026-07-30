import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcFeRows

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.rows_cost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.rows_cost

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.auxiliary_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.auxiliary_count

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.accepted_splits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.accepted_splits

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows.accepted_of_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeRows.accepted_of_rows

end NightstreamTests.Axioms.CanonicalKSplitNcFeRows
