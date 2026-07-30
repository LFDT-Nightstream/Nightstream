import Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the adv-presence boundary.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalAdvPresenceBoundary

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.recomposeAdv_absent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.recomposeAdv_absent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.recomposeAdv_mixed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.recomposeAdv_mixed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.recomposeAdv_present' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.recomposeAdv_present

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.plain_profile_is_structural' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.plain_profile_is_structural

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.presence_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.presence_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.presence_lt_of_absent' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.presence_lt_of_absent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.presence_ne_length_of_absent' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.presence_ne_length_of_absent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary.mixer_unreached_unless_all_present' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AdvPresenceBoundary.mixer_unreached_unless_all_present

end NightstreamTests.Axioms.CanonicalAdvPresenceBoundary
