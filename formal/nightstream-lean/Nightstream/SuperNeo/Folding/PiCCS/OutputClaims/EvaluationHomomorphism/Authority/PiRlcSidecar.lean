import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.Finite

/-!
Conditional authority theorem for the packed `Pi_RLC` `yZcol` sidecar.

Protocol: SuperNeo `Pi_CCS` output followed by `Pi_RLC`.
Phase: source sidecars to one combined parent sidecar.
Constraint family: aggregate sidecar equation and parent semantic matches;
this file emits no rows.

Assurance tier: model-level.

Owns: exact predicates for sourcewise packed binding and canonical aggregate
equality; proof that source binding or the named collision is exactly that one
equality; the modeled aggregate equation, combined-parent projection match,
parent-assignment match; and derivation of the equality from those links.

Does not own: construction of the parent anchor, commitment binding, a
`PiDEC` opening bridge, challenge sampling or validity, a probability bound
for the mixing collision, transcript ordering, Rust/R1CS refinement, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: no digest or aggregate is authority by itself. The result
requires two semantic parent equalities. A later physical refinement must
independently justify the parent opening and prove that these equalities are
what Rust/R1CS checks. Outside those premises, no sourcewise conclusion is
claimed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.sources` | every claimed source sidecar equals its complete-assignment packed projection | target relation | `SourceBound` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.aggregate` | combined claimed sidecar is the exact finite challenge fold of source claims | checked premise | `AggregateEquation` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_projection` | combined claimed sidecar equals the packed projection of the proposed parent assignment | semantic match | `ParentProjectionMatches` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_assignment` | that parent assignment equals the canonical finite `Pi_RLC` fold | semantic match | `ParentAssignmentMatches` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.canonical_aggregate` | claimed and source-derived vectors have the same sampled aggregate | exact local obligation | `CanonicalAggregateEquality` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.collision` | unequal source vectors have the same aggregate under the supplied challenges | security boundary | `MixingCollision` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.exact` | source binding or collision iff canonical aggregate equality | exact model theorem | `sourceBound_or_mixingCollision_iff_aggregateEquality` |
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

/-- The single combined public sidecar equals the projection of the proposed
complete parent assignment at the supplied block point. This is not opening
evidence. -/
def ParentProjectionMatches
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (parentAssignment : SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (combinedClaim : RingK) : Prop :=
  combinedClaim = packedYZcol covers parentAssignment point

/-- The proposed parent assignment equals the exact canonical `Pi_RLC` fold
of the source assignments. This is not commitment-opening evidence. -/
def ParentAssignmentMatches
    {shape : SemanticShape}
    {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (parentAssignment : SemanticAssignment shape) : Prop :=
  parentAssignment = PiRLCFinite.Raw.combineAssignments challenges assignments

/-- The irreducible local equality: claimed and source-derived sidecar vectors
have the same `Pi_RLC` aggregate under the supplied challenges. Whether its two sides are
checked or computed is a later physical ownership decision. -/
def CanonicalAggregateEquality
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (claims : Fin count -> RingK) : Prop :=
  PiRLCFinite.combineEvaluation challenges claims =
    PiRLCFinite.combineEvaluation challenges fun source =>
      packedYZcol covers (assignments source) point

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
    CanonicalAggregateEquality covers challenges assignments point claims

/-- Sourcewise truth or the named mixing collision is exactly one canonical
aggregate equality. This is the semantic obligation ledger; the three exposed
parent links below are only one possible derivation of it. -/
theorem sourceBound_or_mixingCollision_iff_aggregateEquality
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {count : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (claims : Fin count -> RingK) :
    SourceBound covers assignments point claims ∨
        MixingCollision covers challenges assignments point claims <->
      CanonicalAggregateEquality covers challenges assignments point claims := by
  constructor
  · intro outcome
    rcases outcome with sourceBound | collision
    · have claimsEqual : claims = fun source =>
          packedYZcol covers (assignments source) point := by
        funext source
        exact sourceBound source
      simpa [CanonicalAggregateEquality, claimsEqual]
    · exact collision.2
  · intro aggregateEquality
    by_cases sourceBound : SourceBound covers assignments point claims
    · exact Or.inl sourceBound
    · exact Or.inr ⟨sourceBound, aggregateEquality⟩

/-- One exact aggregate equation plus the two parent semantic matches suffice
for all source sidecars, except when the supplied
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
      ParentProjectionMatches covers parentAssignment point combinedClaim)
    (parentAssignmentMatches :
      ParentAssignmentMatches challenges assignments parentAssignment) :
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
        rw [parentAssignmentMatches]
      _ = PiRLCFinite.combineEvaluation challenges fun source =>
          packedYZcol covers (assignments source) point :=
        PackedBlockAction.Finite.packedYZcol_combine
          covers challenges assignments point
  exact (sourceBound_or_mixingCollision_iff_aggregateEquality covers
    challenges assignments point claims).2 aggregateEquality

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiRlcSidecar
