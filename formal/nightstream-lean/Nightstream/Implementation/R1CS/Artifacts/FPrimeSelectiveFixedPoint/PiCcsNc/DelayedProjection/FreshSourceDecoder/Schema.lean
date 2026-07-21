import Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Schema

/-!
Artifact-layer facade for the neutral fresh public-`X` decoder schema.

Owns: the stable artifact-layer import path used by generated shards.

Does not own: generated data, coordinate values, theorems, protocol
authority, or rows.

Emits constraints: none.

| Child path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| neutral `FreshSourceDecoder.Schema` | proof-free record language | model schema | neutral schema |
-/
