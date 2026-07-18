import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PiDecRecomposition

/-!
Curated authority bridges for output-evaluation homomorphisms.

Assurance tier: model-level.

Owns: the single combined-parent `PiDEC` opening dichotomy and its 54-lane
`yZcol` consequence; and the conditional packed-sidecar reduction from one
aggregate equation plus two semantic parent matches to source binding or mixing
collision.

Does not own: child-opening extraction, collision hardness or probability,
NIFS closure, transcripts, Rust/R1CS, or row removal.

Emits constraints: no.

Authority boundary: valid CE openings and accepted public recomposition imply
assignment equality only outside an explicit parent-opening binding collision.
The generic packed pair theorem keeps physical authority explicit. The
optional `PiDecRecomposition` route derives a parent projection equality only
from exact child-sidecar matches and one accepted degree-53 identity; physical
child opening remains a separate refinement.

| Stage path | Child owner | Guarantee | Excluded boundary |
|---|---|---|---|
| `nifs.pi_dec.verify.authority.parent_opening.binding` | `PiDECParentOpening` | combined parent equals production radix recomposition or exposes a collision | collision hardness and production refinement |
| `nifs.pi_dec.verify.authority.parent_opening.y_zcol` | `PiDECParentOpening` | all 54 combined-parent lanes transport under the same dichotomy | collision hardness and production refinement |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.reduce` | `PiRlcSidecar` | aggregate equation plus parent projection/assignment matches yields source binding or a named mixing collision | opening/recomputation refinement and collision bound |
| `nifs.shared.delayed_packed.pair` | `DelayedPackedProjection` | one fixed-width pair is exact or exposes a degree-53 bad root | physical source binding and bad-root bound |
| `nifs.pi_dec.delayed_packed.parent_projection` | `DelayedPackedProjection.PiDecRecomposition` | an accepted child recomposition derives the parent equality or a degree-53 bad root | child extraction, transcript timing, and bad-root bound |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.pi_dec_route` | `DelayedPackedProjection.PiDecRecomposition` | source binding, mixing collision, or projection bad root | child extraction and both probability bounds |
| `nifs.pi_rlc.verify.identities.y_zcol.recombine` | `DelayedPackedProjection.LimbDecomposition` | the two base-limb evaluations recombine exactly to the packed parent projection | generated-column and transcript refinement |
-/
