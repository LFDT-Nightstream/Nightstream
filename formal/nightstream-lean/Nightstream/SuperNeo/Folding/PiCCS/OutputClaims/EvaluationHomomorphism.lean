import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority

/-!
Curated homomorphism surface for source-derived `Pi_CCS` output sidecars.

Owns: the closed base-field-linear `yZcol` interface proved by `BaseLinear`
and the model-level combined-parent opening dichotomy exported by `Authority`.

Does not own: the upstream `PiRLC` `RingF` action from source claims to its
combined parent, binding-collision hardness, NIFS closure, transcript
derivation, Rust/R1CS correspondence, or row removal.

Emits constraints: no.

Authority boundary: the sourcewise transport theorem is algebraic and requires
recomposition equality as a premise. Separately, valid parent/child CE openings
and `PiDEC.Accepted` derive that equality for the one combined parent, or expose
an explicit parent-opening binding collision. Neither result connects the
`PiCCS` source product to that parent.

| Stage path | Child owner | Mathematical obligation | Status |
|---|---|---|---|
| `nifs.pi_ccs.output.y_zcol.projection` | `BaseLinear` | zero, add, and base-`F` scaling preserve the canonical projection | proved |
| `nifs.pi_dec.verify.recomposition.y_zcol` | `BaseLinear.yZcolEvaluation_piDecRecompose` | all 54 lanes use verifier-fixed radix weights | proved |
| `nifs.pi_ccs.output.y_zcol.authority` | `BaseLinear.canonicalYZcol_product_piDec_transport` | sourcewise algebraic transport from explicitly recomposed assignments | conditional utility; not production protocol order |
| `nifs.pi_dec.verify.authority.parent_opening` | `Authority.PiDECParentOpening` | one valid combined parent equals child recomposition or exposes a binding collision | proved model-level dichotomy |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol` | not present | source-product claims determine the combined parent sidecar | open |
-/
