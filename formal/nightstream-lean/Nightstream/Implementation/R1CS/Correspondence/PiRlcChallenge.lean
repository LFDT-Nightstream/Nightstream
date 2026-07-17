import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler

/-! Public ownership boundary for Pi_RLC challenge correspondence.

Owns: the concrete transcript machine, transcript schedules, sampler equation
families, and their current local refinement results.

Does not own: Pi_RLC ring verification, complete NIFS/F′ refinement, full-F′
cost totals, or security bounds.

Emits constraints: no.

| Phase | Mathematical obligation | Excluded boundary |
|---|---|---|
| `TranscriptMachine` | executable verifier-owned Poseidon2 state machine | permutation security |
| `Transcript` | recursive/terminal schedules refine that machine | candidate validity |
| `Sampler` | challenge candidates and selection refine typed outputs | Pi_RLC algebra and outer placement |
-/
