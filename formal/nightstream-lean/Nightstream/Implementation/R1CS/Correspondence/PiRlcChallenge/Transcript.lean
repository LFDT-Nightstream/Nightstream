import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ChunkOrder
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.OutputDigestSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.PinSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Schedule
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ScheduleRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal

/-! Parent for Pi_RLC challenge transcript correspondence.

Owns: chunk ordering, verifier-owned digest semantics, digest/pin schedules,
recursive schedule refinement, and the terminal-profile subtree.

Does not own: the underlying Poseidon2 permutation proof, candidate range
semantics, or the Pi_RLC algebra checks.

Emits constraints: no.

| Child family | Mathematical obligation | Excluded boundary |
|---|---|---|
| chunk order | deterministic candidate absorb order | sampler acceptance |
| digest/pins/schedule | verifier-owned transcript state and challenges | Poseidon2 security |
| schedule refinement | concrete schedule matches transcript machine | outer NIFS composition |
| terminal | terminal-profile schedule and output digest | recursive profile |
-/
