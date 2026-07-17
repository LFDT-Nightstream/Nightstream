import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar

/-!
Curated authority bridges for output-evaluation homomorphisms.

Owns: the single combined-parent `PiDEC` opening dichotomy and its 54-lane
`yZcol` consequence; and the conditional packed-sidecar reduction from one
aggregate equation plus one parent anchor to source binding or mixing
collision.

Does not own: construction of the packed parent anchor, collision hardness or
probability, NIFS closure, transcripts, Rust/R1CS, or row removal.

Emits constraints: no.

Authority boundary: valid CE openings and accepted public recomposition imply
assignment equality only outside an explicit parent-opening binding collision.
Separately, the packed-sidecar theorem assumes—not derives—the combined-parent
projection anchor and assignment binding.

| Stage path | Child owner | Guarantee | Excluded boundary |
|---|---|---|---|
| `nifs.pi_dec.verify.authority.parent_opening.binding` | `PiDECParentOpening` | combined parent equals production radix recomposition or exposes a collision | collision hardness and production refinement |
| `nifs.pi_dec.verify.authority.parent_opening.y_zcol` | `PiDECParentOpening` | all 54 combined-parent lanes transport under the same dichotomy | collision hardness and production refinement |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.reduce` | `PiRlcSidecar` | aggregate equation plus parent projection/assignment binding yields source binding or a named mixing collision | parent-anchor construction and collision bound |
-/
