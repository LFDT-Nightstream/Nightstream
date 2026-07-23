import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.RoundMaps

/-!
Stable artifact facade for generated combined-NC round column maps.

Owns: the handwritten import boundary for the fixed ordered map from each
round-local slot to source columns.

Does not own: decoder correctness, round-polynomial semantics, row
satisfaction, transcript sampling, costs, or row-removal authority.

| Stable stage path | Generated owner | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.round_maps` | `Generated.RoundMaps` | computed artifact |
-/
