import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.ChallengeWiring
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.SamplerLayout
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.TranscriptLayout

/-!
Stable artifact surface for the emitted three-matrix diagnostic PiRLC challenge phase.

| Child | Evidence | Remaining boundary |
|---|---|---|
| `ChallengeWiring` | all 810 selected coefficient outputs are the exact rho coefficient consumers | sampler-row semantics and transcript authority |
| `SamplerLayout` | typed affine layout, block-major lane predecessors, and exact output aliases for the active 15-scalar sampler | row satisfaction, sampler semantics, and transcript authority |
| `TranscriptLayout` | exact 291-pin / 78-call source partition, ordered emissions, state continuity, 240 field-output aliases, and four external bind-input columns | row satisfaction, transcript replay, and transcript authority |
-/
