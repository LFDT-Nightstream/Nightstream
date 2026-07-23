import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows

/-!
Stable aggregate artifact facade for all combined-NC source rows.

Owns: the handwritten import boundary for the complete ordered concatenation
of the 63 generated source-row shards. Consumers needing only a phase range
should use the narrower sibling facades.

Does not own: coefficient decoding, row satisfaction, compiler semantics,
acceptance, costs, or row-removal authority.

| Stable stage path | Generated owner | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_rows.all` | `Generated.SourceRows` | computed artifact |
-/
