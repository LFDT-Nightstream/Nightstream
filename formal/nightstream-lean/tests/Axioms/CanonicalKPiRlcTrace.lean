import Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKPiRlcTrace

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.trace_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.trace_valid

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.trace_pairs_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.trace_pairs_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.traces_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.traces_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.occurrence_rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.occurrence_rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.occurrence_exact_or_badRoot' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.occurrence_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.occurrence_rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcTrace.occurrence_rows_honest

end NightstreamTests.Axioms.CanonicalKPiRlcTrace
