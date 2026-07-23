import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk2

/-!
Stable bounded artifact facade for the padding source-row prefix.

Owns: the handwritten import boundary for generated source-row chunks 0--2,
the smallest shard cone containing the padding obligations.

Does not own: padding semantics, row satisfaction, selector truth, costs, or
row-removal authority.

| Stable stage path | Generated owners | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_rows.padding` | `Generated.SourceRows.Chunk0`--`Chunk2` | computed artifact |
-/
