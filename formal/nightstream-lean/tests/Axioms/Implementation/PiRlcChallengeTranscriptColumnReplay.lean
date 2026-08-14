import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay
import tests.Axioms.Support

/-! Fail-closed axiom guard for exact Rust slice-boundary replay. -/

namespace NightstreamTests.Axioms.Implementation.PiRlcChallengeTranscriptColumnReplay

open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay.normalizeSlice_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms normalizeSlice_sound

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay.executeSlice_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms executeSlice_sound

end NightstreamTests.Axioms.Implementation.PiRlcChallengeTranscriptColumnReplay
