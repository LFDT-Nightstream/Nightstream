import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening

/-!
Curated authority bridges for output-evaluation homomorphisms.

Owns: the single combined-parent `PiDEC` opening dichotomy and its 54-lane
`yZcol` consequence.

Does not own: the upstream `PiRLC` action from the `PiCCS` source product to the
combined parent, collision hardness, NIFS closure, transcripts, Rust/R1CS, or
row removal.

Emits constraints: no.

Authority boundary: valid CE openings and accepted public recomposition imply
assignment equality only outside an explicit parent-opening binding collision.

| Stage path | Child owner | Guarantee | Status |
|---|---|---|---|
| `nifs.pi_dec.verify.authority.parent_opening.binding` | `PiDECParentOpening` | combined parent equals production radix recomposition or exposes a collision | model-level dichotomy |
| `nifs.pi_dec.verify.authority.parent_opening.y_zcol` | `PiDECParentOpening` | all 54 combined-parent lanes transport under the same dichotomy | model-level dichotomy |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol` | not present | connect source-product claims to the combined parent | open |
-/
