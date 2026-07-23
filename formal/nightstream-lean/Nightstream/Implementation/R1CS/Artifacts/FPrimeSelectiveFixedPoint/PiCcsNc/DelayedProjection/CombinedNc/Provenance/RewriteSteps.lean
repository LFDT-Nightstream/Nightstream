import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.RewriteSteps

/-!
Stable narrow artifact facade for generated rewrite-step shards.

Owns: the handwritten import boundary for the ordered rewrite-step list used
by bounded batch-index certificates, without importing unrelated provenance
families.

Does not own: rewrite semantics, source satisfaction, selected-row
satisfaction, costs, or row-removal authority.

| Stable stage path | Generated owner | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.provenance.rewrites` | `Generated.Provenance.RewriteSteps` | computed artifact |
-/
