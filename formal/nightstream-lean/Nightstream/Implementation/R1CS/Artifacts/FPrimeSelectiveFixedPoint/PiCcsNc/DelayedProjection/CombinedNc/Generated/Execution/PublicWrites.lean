/-
Generated file: production combined-NC execution artifact; do not hand-edit.

Owns: the exact ordered concatenation of 2 bounded PublicWrites shards.

Does not own: row satisfaction, commitment binding, semantic acceptance,
security reductions, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.execution` | The generated execution payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites.Chunk1

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites

def values : List RawPublicWrite :=
  Chunk0.values ++
  Chunk1.values

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.PublicWrites
