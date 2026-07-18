import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity

/-!
Curated homomorphism surface for source-derived `Pi_CCS` output sidecars.

Assurance tier: model-level.

Owns: the closed base-field-linear flat `yZcol` diagnostics proved by
`BaseLinear`; the model-level combined-parent opening dichotomy and conditional
packed-sidecar collision reduction exported by `Authority`; and the
counterexamples showing why the flat-assignment projection cannot commute with
the current ring action and why the source scalar must be bound to the
canonical parent projection.

Does not own: physical justification of the packed combined-parent projection,
binding- or mixing-collision hardness, NIFS closure, transcript derivation,
Rust/R1CS correspondence, or row removal.

Emits constraints: no.

Authority boundary: the sourcewise transport theorem is algebraic and requires
recomposition equality as a premise. Separately, valid parent/child CE openings
and `PiDEC.Accepted` derive that equality for the one combined parent, or expose
an explicit parent-opening binding collision. Neither result connects the
`PiCCS` source product to that parent.

The packed block projection is exported only through a conditional theorem:
it requires two semantic parent equalities, then leaves a named mixing
collision. A later refinement must justify those equalities from authoritative
data. This is not accepted production authority or permission to remove rows.

| Stage path | Child owner | Mathematical guarantee | Excluded boundary |
|---|---|---|---|
| `nifs.pi_ccs.output.y_zcol.projection` | `BaseLinear` | zero, add, and base-`F` scaling preserve the canonical projection | concrete carrier/Rust refinement |
| `nifs.pi_dec.verify.recomposition.y_zcol` | `BaseLinear.yZcolEvaluation_piDecRecompose` | all 54 lanes use verifier-fixed radix weights | concrete carrier/Rust refinement |
| `nifs.pi_ccs.output.y_zcol.authority` | `BaseLinear.canonicalYZcol_product_piDec_transport` | sourcewise algebraic transport from explicitly recomposed assignments | production protocol-order authority |
| `nifs.pi_dec.verify.authority.parent_opening` | `Authority.PiDECParentOpening` | one valid combined parent equals child recomposition or exposes a binding collision | collision hardness and production refinement |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.reduce` | `Authority.PiRlcSidecar` | aggregate plus parent semantic matches implies sourcewise binding or a named mixing collision | opening/recomputation refinement, transcript sampling, and collision bound |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol.necessity` | `Necessity.FlatColumnAction` | flat-assignment projection does not commute with the ring action at arbitrary column points | expanded-witness semantics required |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.scalar_match.necessity` | `Necessity.ScalarBinding` | a source-only scalar equality admits a forged claim outside mixing and bad-root events | independently justified parent equality required |
-/
