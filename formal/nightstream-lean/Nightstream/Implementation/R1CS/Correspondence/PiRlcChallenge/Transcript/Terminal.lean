import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.PinSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.ScheduleRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.CandidateRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestPins
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestBinding

/-! Parent for terminal-profile Pi_RLC transcript correspondence.

Owns: terminal schedule, pins, digest rounds, candidate replay, output-digest
schedule, and their local refinement/binding results.

Does not own: the recursive-profile schedule, sampler rows, or Poseidon2
collision resistance.

Emits constraints: no.

| Child family | Mathematical obligation | Excluded boundary |
|---|---|---|
| schedule/pins/digest rounds | exact terminal transcript execution | recursive profile |
| candidate refinement | candidate stream agrees with terminal transcript | selection rows |
| output digest | final digest schedule, pins, and binding | digest collision bound |
-/
