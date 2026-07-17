import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker

/-!
Canonical fixed-active NIFS verifier surface.

Owns: the verifier-constructed `1 + 14` carrier, the four retained incoming
parent equations, and the exact raw-certificate checker.

Does not own: semantic source truth, bad-event bounds, Rust/R1CS refinement,
physical rows, costs, or row removal.

Emits constraints: no.

| Child | Mathematical ownership | Assurance boundary |
|---|---|---|
| `Context` | derive structure, stages, parent presence, and source consistency | exact model construction |
| `RunningAuthority` | check point and three `Pi_DEC` recomposition families | exact model equivalence |
| `Checker` | compose incoming authority, `Pi_CCS`, sampler, and outgoing `Pi_DEC` | exact physical NIFS acceptance |
-/
