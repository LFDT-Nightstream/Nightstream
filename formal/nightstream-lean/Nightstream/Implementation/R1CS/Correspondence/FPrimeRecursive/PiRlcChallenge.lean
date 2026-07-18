import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.ProjectionConsumer
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Transcript.Handoff

/-!
Stable correspondence surface for the emitted three-matrix diagnostic PiRLC challenge.

| Child | Result | Remaining boundary |
|---|---|---|
| `ProjectionConsumer` | zero-cost 15 x 54 sampler-output alias to projection rho inputs | active sampler rows, transcript replay, and post-PiCCS state authority |
| `Sampler` | exact source-leaf embedding to independent field-derived 54-of-64 semantics | Poseidon2 replay/chaining, ring assembly, and post-PiCCS state authority |
| `Transcript.Handoff` | embedded Poseidon2 rows plus sampler rows derive all 15 projection challenges | post-PiCCS state/digest authority, complete Rust-row identity, and security bounds |
-/
