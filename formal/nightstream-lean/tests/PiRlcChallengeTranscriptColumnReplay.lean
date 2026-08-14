import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay

/-! Focused checks for exact Rust slice-boundary transcript replay. -/

namespace tests.PiRlcChallengeTranscriptColumnReplay

open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay

#check normalizeSlice_sound
#check executeSlice_sound

end tests.PiRlcChallengeTranscriptColumnReplay
