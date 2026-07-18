import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Linear

/-!
Contract: transport packed block projection through the exact base-field
recomposition used by production-parameter Π_DEC.

Owns: generic finite base-field combination and its exact `b = 2`, `k = 14`
Π_DEC specialization for `packedYZcol`.

Does not own: digit splitting, child norm bounds, augmented CE membership,
parent/child acceptance, commitment binding, transcripts, Rust, R1CS, costs,
or row removal.

Emits constraints: no.

Authority boundary: assignments, fixed weights, and the evaluation point are
explicit typed inputs. The theorem proves algebraic recomposition only; it
does not prove that any Π_DEC message carries or binds this evaluation.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.recomposition.packed_y_zcol.combine` | every packed lane uses the same finite base-field weights as the complete assignment | derived | `packedYZcol_baseCombine` |
| `nifs.pi_dec.verify.recomposition.packed_y_zcol.radix` | specialize to verifier-fixed production radix weights | computed | `packedYZcol_piDecRecompose` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.PiDEC

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open PackedBlockAction
open Linear

/-- Exact finite base-field combination of packed evaluations. -/
theorem packedYZcol_baseCombine
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (weights : Fin count -> F)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers
        (Phi81Relation.EvaluationHomomorphism.BaseLinear.Raw.combineAssignments
          weights assignments) point =
      combineEvaluations weights fun index =>
        packedYZcol covers (assignments index) point := by
  induction count with
  | zero => exact packedYZcol_zero covers point
  | succ count inductionHypothesis =>
      rw [Phi81Relation.EvaluationHomomorphism.BaseLinear.Raw.combineAssignments,
        combineEvaluations,
        packedYZcol_add,
        packedYZcol_scale, inductionHypothesis]
      rfl

/-- Exact production specialization: packed evaluation uses the identical
`b = 2`, `k = 14` weights as Π_DEC assignment recomposition. -/
theorem packedYZcol_piDecRecompose
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (assignments : Fin productionGlobalParams.k ->
      SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables) :
    packedYZcol covers
        (Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
          assignments)
        point =
      combineEvaluations
        Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => packedYZcol covers (assignments index) point) := by
  unfold Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
  exact packedYZcol_baseCombine covers
    Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight assignments point

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.PiDEC
