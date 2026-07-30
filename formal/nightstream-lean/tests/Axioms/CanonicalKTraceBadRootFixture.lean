import Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKTraceBadRootFixture

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.trace_badRoot' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.trace_badRoot

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.occurrence_rows_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.occurrence_rows_satisfied

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.occurrence_badRoot_of_satisfied_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.occurrence_badRoot_of_satisfied_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.not_eventFreeOccurrenceSoundness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.not_eventFreeOccurrenceSoundness

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.occurrence_badRoot_at_source' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.occurrence_badRoot_at_source

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture.occurrence_not_exact_at_source' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceBadRootFixture.occurrence_not_exact_at_source

end NightstreamTests.Axioms.CanonicalKTraceBadRootFixture
