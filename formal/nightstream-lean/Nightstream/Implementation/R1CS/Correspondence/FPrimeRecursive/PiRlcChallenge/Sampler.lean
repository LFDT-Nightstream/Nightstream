import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.FirstAccepted

/-!
Stable active PiRLC sampler refinement surface.

| Child | Owns | Stops before |
|---|---|---|
| `Rows` | normalized source slices, transcript/sampler leaf separation, exact source-row counts | mathematical sampler meaning |
| `ScalarSemantics` | 16-lane accept/symbol/count chain for each scalar | selection tail and transcript provenance |
| `TailSources` | 64 lane-to-tail inputs, distinct zero initializers, 54 output aliases | first-accepted semantics |
| `FirstAccepted` | bounded success and all 810 field-derived output values | Poseidon2 provenance and ring assembly |

No child treats the historical circuit, a digest, or a measured count as
semantic authority.
-/
