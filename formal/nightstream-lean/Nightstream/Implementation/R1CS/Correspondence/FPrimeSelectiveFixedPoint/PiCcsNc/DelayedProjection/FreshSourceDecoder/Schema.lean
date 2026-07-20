import Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Schema

/-!
Correspondence-layer facade for the neutral fresh public-`X` decoder schema.

Owns: the stable correspondence-layer import path for the neutral schema.

Does not own: generated data, value equations, protocol authority, or rows.

Emits constraints: none.

| Child path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| neutral `FreshSourceDecoder.Schema` | shared typed record language | model schema | neutral schema |
-/
