import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk5

/-!
Stable bounded artifact facade for initial-claim source rows.

Owns: the handwritten import boundary for generated source-row chunks 2--5,
the smallest shard cone containing the initial combined-NC equation.

Does not own: initial-claim semantics, row satisfaction, transcript authority,
costs, or row-removal authority.

| Stable stage path | Generated owners | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_rows.initial` | `Generated.SourceRows.Chunk2`--`Chunk5` | computed artifact |
-/
