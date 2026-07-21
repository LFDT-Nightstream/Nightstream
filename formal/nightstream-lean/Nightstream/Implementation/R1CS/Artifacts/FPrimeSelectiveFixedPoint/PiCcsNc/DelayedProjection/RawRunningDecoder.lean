import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Exact

/-!
Public artifact facade for the fixed-profile running-`X` public-prefix
decoder.

This facade owns only the generated `14 × 270` column map, its bounded census,
and exact coordinate/physical-column ownership. Assignment semantics and
protocol acceptance live in the correspondence layer.

Owns: the public artifact interface for the fixed-profile running-`X` decoder.

Does not own: full `CcsWitness.Z`/`CeWitness.Z` assignment semantics, R1CS
satisfaction, protocol acceptance, transcript scheduling, or commitment
authority.

Emits constraints: none; artifact facade only.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact` | expose the exact generated decoder census and ownership facts | checked artifact |
-/
