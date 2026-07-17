import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Replay

/-! Focused checks for complete typed terminal-NC replay. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check Replay.messages
#check Replay.rawRounds_eq_messages
#check Replay.laterFinalCall_eq_finalArtifact
#check Replay.runRounds_eq_finalCallOutput
#check Replay.afterNc_eq_finalCallOutput
#check Replay.finalPermutationBound_of_exactSchedule
