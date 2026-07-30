import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcNcRows

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcRows.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows.rows_cost' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcRows.rows_cost

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows.auxiliary_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcNcRows.auxiliary_count

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows.rows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcRows.rows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows.accepted_of_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcRows.accepted_of_rows

end NightstreamTests.Axioms.CanonicalKSplitNcNcRows
