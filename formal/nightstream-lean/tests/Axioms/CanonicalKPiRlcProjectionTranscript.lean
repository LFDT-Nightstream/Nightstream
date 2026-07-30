import Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the transcript-bound public PiRLC projection
occurrence.
-/

namespace NightstreamTests.Axioms.CanonicalKPiRlcProjectionTranscript

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript.replay_entries_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPiRlcProjectionTranscript.replay_entries_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript.projectionColumns_belowBase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcProjectionTranscript.projectionColumns_belowBase

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript.beta_eq_transcript' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcProjectionTranscript.beta_eq_transcript

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPiRlcProjectionTranscript.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript.equations_or_transcriptBadRoot_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  KPiRlcProjectionTranscript.equations_or_transcriptBadRoot_of_rows

end NightstreamTests.Axioms.CanonicalKPiRlcProjectionTranscript
