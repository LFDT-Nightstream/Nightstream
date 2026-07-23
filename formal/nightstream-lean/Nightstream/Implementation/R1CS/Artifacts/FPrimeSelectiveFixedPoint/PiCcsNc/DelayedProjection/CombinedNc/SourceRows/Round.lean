import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk9
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk10
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.SourceRows.Chunk11

/-!
Stable bounded artifact facade for per-round source rows.

Owns: the handwritten import boundary for generated source-row chunks 5--11,
the smallest shard cone containing every fixed combined-NC round equation.

Does not own: round-polynomial semantics, row satisfaction, challenge
sampling, costs, or row-removal authority.

| Stable stage path | Generated owners | Authority class |
Emits constraints: no.

|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_rows.rounds` | `Generated.SourceRows.Chunk5`--`Chunk11` | computed artifact |
-/
