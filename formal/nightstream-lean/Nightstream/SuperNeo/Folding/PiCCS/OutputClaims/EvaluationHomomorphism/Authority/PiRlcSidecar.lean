import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Finite

/-!
Conditional authority theorem for the packed `Pi_RLC` `yZcol` sidecar.

Protocol: SuperNeo `Pi_CCS` output followed by `Pi_RLC`.
Phase: source sidecars to one combined parent sidecar.
Constraint family: aggregate sidecar equation and parent projection anchor;
this file emits no rows.

Owns: exact predicates for sourcewise packed binding, the physical aggregate
equation, the combined-parent projection anchor, parent-assignment binding,
and the theorem reducing all sourcewise disagreement to one named `Pi_RLC`
mixing collision.

Does not own: construction of the parent anchor, commitment binding, a
`PiDEC` opening bridge, challenge sampling or validity, a probability bound
for the mixing collision, transcript ordering, Rust/R1CS refinement, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: no digest or aggregate is authority by itself. The result
requires both an independently justified parent projection and equality of the
opened parent assignment with the canonical `Pi_RLC` assignment fold. Outside
those premises, no sourcewise conclusion is claimed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.sources` | every claimed source sidecar equals its complete-assignment packed projection | target relation | `SourceBound` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.aggregate` | combined claimed sidecar is the exact finite challenge fold of source claims | checked premise | `AggregateEquation` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_projection` | combined claimed sidecar equals the packed projection of one opened parent assignment | security premise | `ParentProjectionAnchor` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_assignment` | that parent assignment is the canonical finite `Pi_RLC` fold | binding premise | `ParentAssignmentBound` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.collision` | unequal source vectors have the same verifier-sampled aggregate | security boundary | `MixingCollision` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.reduce` | the three premises imply source binding or the named collision | derived | `sourceBound_or_mixingCollision` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open PackedBlockAction

/-- Every source claim is the packed block-domain projection of the same
complete assignment that enters the canonical `Pi_RLC` fold. -/
def SourceBound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (claims : Fin count -> RingK) : Prop :=
  forall source,
    claims source = packedYZcol covers (assignments source) point

/-- The physical sidecar equation checked by `Pi_RLC`: the combined public
claim is the exact finite challenge fold of the source claims. -/
def AggregateEquation
    {count : Nat}
    (challenges : Fin count -> RingF)
    (claims : Fin count -> RingK)
    (combinedClaim : RingK) : Prop :=
  combinedClaim = PiRLCFinite.combineEvaluation challenges claims

/-- The single combined public sidecar is anchored to one independently
opened complete parent assignment at the verifier-owned block point. -/
def ParentProjectionAnchor
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (parentAssignment : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (combinedClaim : RingK) : Prop :=
  combinedClaim = packedYZcol covers parentAssignment point

/-- The independently opened parent assignment is the exact canonical
`Pi_RLC` fold of the source assignments. -/
def ParentAssignmentBound
    {shape : SemanticShape}
    {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (parentAssignment : SemanticAssignment shape) : Prop :=
  parentAssignment = PiRLCFinite.Raw.combineAssignments challenges assignments

/-- Named security event: at least one source sidecar is false, but the false
and canonical source vectors have the same sampled `Pi_RLC` aggregate. -/
def MixingCollision
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (claims : Fin count -> RingK) : Prop :=
  ¬ SourceBound covers assignments point claims ∧
    PiRLCFinite.combineEvaluation challenges claims =
      PiRLCFinite.combineEvaluation challenges fun source =>
        packedYZcol covers (assignments source) point

/-- One exact aggregate equation plus one parent projection anchor and parent
assignment binding suffice for all source sidecars, except when the sampled
`Pi_RLC` combination collides on two distinct source vectors. -/
theorem sourceBound_or_mixingCollision
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (claims : Fin count -> RingK)
    (parentAssignment : SemanticAssignment shape)
    (combinedClaim : RingK)
    (aggregate : AggregateEquation challenges claims combinedClaim)
    (parentProjection :
      ParentProjectionAnchor covers parentAssignment point combinedClaim)
    (parentAssignmentBound :
      ParentAssignmentBound challenges assignments parentAssignment) :
    SourceBound covers assignments point claims ∨
      MixingCollision covers challenges assignments point claims := by
  have aggregateEquality :
      PiRLCFinite.combineEvaluation challenges claims =
        PiRLCFinite.combineEvaluation challenges fun source =>
          packedYZcol covers (assignments source) point := by
    calc
      PiRLCFinite.combineEvaluation challenges claims = combinedClaim :=
        aggregate.symm
      _ = packedYZcol covers parentAssignment point := parentProjection
      _ = packedYZcol covers
          (PiRLCFinite.Raw.combineAssignments challenges assignments) point := by
        rw [parentAssignmentBound]
      _ = PiRLCFinite.combineEvaluation challenges fun source =>
          packedYZcol covers (assignments source) point :=
        PackedBlockAction.Finite.packedYZcol_combine
          covers challenges assignments point
  by_cases sourceBound : SourceBound covers assignments point claims
  · exact Or.inl sourceBound
  · exact Or.inr ⟨sourceBound, aggregateEquality⟩

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar
