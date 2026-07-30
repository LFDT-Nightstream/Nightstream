import Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the transcript-mode boundary.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalTranscriptModeBoundary

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary.modes_agree_on_initial_state' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptModeBoundary.modes_agree_on_initial_state

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary.modes_differ' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms TranscriptModeBoundary.modes_differ

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary.divergent_values' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms TranscriptModeBoundary.divergent_values

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary.sponge_is_fixed_arity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TranscriptModeBoundary.sponge_is_fixed_arity

end NightstreamTests.Axioms.CanonicalTranscriptModeBoundary
