import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Authority

/-!
Opening-derived fixed-active NIFS carrier.

Owns: the small facade joining opening-derived materialization and authority.

Does not own: either child module's equations, physical refinement, costs, or
row removal.

Emits constraints: no.

Authority boundary: importing this facade adds no accepted witness or digest;
all authority remains in the child theorems listed below.

| Child | Mathematical ownership | Assurance boundary |
|---|---|---|
| `Context` | compute the full parent and fourteen children from one opening | exact model construction |
| `Authority` | derive norm, canonicality, and strict PiDEC from validated sources | model-level obligation derivation |

Physical source refinement, an opening-handle serializer, Rust/R1CS lowering,
costs, and row removal remain explicit downstream obligations.
-/
