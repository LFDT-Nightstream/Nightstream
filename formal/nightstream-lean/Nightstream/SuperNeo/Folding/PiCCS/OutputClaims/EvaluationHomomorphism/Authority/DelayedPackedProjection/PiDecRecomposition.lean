import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction.PiDEC

/-!
Optional `Pi_DEC` specialization of the packed delayed projection.

Protocol: SuperNeo `Pi_RLC -> Pi_DEC`.
Phase: compare one packed parent with the radix recomposition of its children.
Constraint family: one 54-coefficient projection identity; this file emits no
rows.

Assurance tier: model-level.

Owns: the child-side radix recomposition, its specialization of the generic
packed pair check, and the conditional source-binding composition that uses
semantically matched child sidecars.

Does not own: the preferred direct-parent opening, child opening extraction,
commitment binding, transcript timing, bad-root or mixing probability,
Poseidon2, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: this is an optional proof route, not the minimal packed
`yZcol` authority boundary. Every child sidecar and the parent/child assignment
recomposition remain explicit premises. A self-consistent child vector or
parent digest is never treated as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.delayed_packed.identity` | compare parent and recomposed-child coefficient vectors | checked | `identity`, `Accepted` |
| `nifs.pi_dec.delayed_packed.scalar` | match one scalar to the child recomposition | semantic match | `ScalarRecompositionMatches` |
| `nifs.pi_dec.delayed_packed.completeness` | exact recomposition accepts at every point | derived | `accepted_of_exact` |
| `nifs.pi_dec.delayed_packed.soundness` | acceptance gives exact equality or a degree-53 bad root | security boundary | `accepted_implies_exact_or_badRoot` |
| `nifs.pi_dec.delayed_packed.parent_projection` | semantically matched children derive the parent projection equality | derived | `accepted_implies_parentProjection_or_badRoot` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.pi_dec_route` | source binding, mixing collision, or projection bad root | derived | `sourceBound_or_mixingCollision_or_badRoot` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PiDecRecomposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-- Canonical production-radix recomposition of 14 packed child sidecars. -/
def recomposeClaims
    (children : Fin productionGlobalParams.k -> RingK) : RingK :=
  BaseLinear.combineEvaluations PiDEC.radixWeight children

/-- One fixed-width parent-versus-children identity. The producer challenge is
explicit so transcript timing cannot be hidden in the semantic definition. -/
def identity
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K) : ProjectionCheck.Identity K :=
  pairIdentity parent (recomposeClaims children) producerBeta

/-- Acceptance surface for the optional child-recomposition route. -/
def Accepted
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K) : Prop :=
  PairAccepted parent (recomposeClaims children) producerBeta

/-- Optional transition check from a computed source aggregate to a proposed
child recomposition. Child-opening authority is a separate premise of later
composition theorems. -/
def TransitionAccepted
    {count : Nat}
    (challenges : Fin count -> RingF)
    (sourceClaims : Fin count -> RingK)
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K) : Prop :=
  Accepted (sourceAggregate challenges sourceClaims) childClaims producerBeta

/-- The claimed scalar equals the canonical radix recomposition of child
sidecars. This is a semantic equality target, not child-opening evidence. -/
def ScalarRecompositionMatches
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (claimedProjection producerBeta : K) : Prop :=
  PairRightScalarMatches (recomposeClaims childClaims) claimedProjection
    producerBeta

/-- The source-scalar check plus child-recomposition equality instantiates the
same degree-53 fixed-width pair identity. -/
theorem transitionAccepted_of_scalar
    {count : Nat}
    (challenges : Fin count -> RingF)
    (sourceClaims : Fin count -> RingK)
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (claimedProjection producerBeta : K)
    (sourceMatches : SourceProjectionMatches challenges sourceClaims
      claimedProjection producerBeta)
    (childrenMatch : ScalarRecompositionMatches childClaims claimedProjection
      producerBeta) :
    TransitionAccepted challenges sourceClaims childClaims producerBeta := by
  exact pairAccepted_of_scalar_matches
    (sourceAggregate challenges sourceClaims) (recomposeClaims childClaims)
    claimedProjection producerBeta sourceMatches childrenMatch

/-- Both coefficient vectors have exactly 54 entries and degree at most 53. -/
theorem identity_wellFormed
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K) :
    (identity parent children producerBeta).WellFormed := by
  exact pairIdentity_wellFormed parent (recomposeClaims children) producerBeta

/-- Exactness is exactly packed parent/child-recomposition equality. -/
theorem identity_exact_iff
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K) :
    (identity parent children producerBeta).Exact <->
      parent = recomposeClaims children := by
  exact pairIdentity_exact_iff parent (recomposeClaims children) producerBeta

/-- Honest completeness: exact packed recomposition accepts at every point. -/
theorem accepted_of_exact
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K)
    (exact : parent = recomposeClaims children) :
    Accepted parent children producerBeta := by
  apply ProjectionCheck.exact_is_accepted
  · exact identity_wellFormed parent children producerBeta
  · exact (identity_exact_iff parent children producerBeta).2 exact

/-- Deterministic soundness for the optional child-recomposition route. -/
theorem accepted_implies_exact_or_badRoot
    (parent : RingK)
    (children : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K)
    (accepted : Accepted parent children producerBeta) :
    parent = recomposeClaims children ∨
      ProjectionCheck.BadRoot projectionOps
        (identity parent children producerBeta) := by
  exact pairAccepted_implies_exact_or_badRoot parent
    (recomposeClaims children) producerBeta accepted

/-- Semantically matched child sidecars plus an accepted check derive the
packed parent projection equality, except at the named bad root. Physical
child-opening authority remains outside this theorem. -/
theorem accepted_implies_parentProjection_or_badRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (parentAssignment : PackedBlockAction.SemanticAssignment shape)
    (childAssignments : Fin productionGlobalParams.k ->
      PackedBlockAction.SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (parentClaim : RingK)
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K)
    (parentRecomposes :
      parentAssignment = PiDEC.Raw.recomposeAssignment childAssignments)
    (childrenMatch : forall child,
      childClaims child =
        PackedBlockAction.packedYZcol covers (childAssignments child) point)
    (accepted : Accepted parentClaim childClaims producerBeta) :
    PiRlcSidecar.ParentProjectionMatches covers parentAssignment point
        parentClaim ∨
      ProjectionCheck.BadRoot projectionOps
        (identity parentClaim childClaims producerBeta) := by
  rcases accepted_implies_exact_or_badRoot parentClaim childClaims
      producerBeta accepted with exact | badRoot
  · left
    unfold PiRlcSidecar.ParentProjectionMatches
    calc
      parentClaim = recomposeClaims childClaims := exact
      _ = BaseLinear.combineEvaluations PiDEC.radixWeight (fun child =>
          PackedBlockAction.packedYZcol covers
            (childAssignments child) point) := by
        apply congrArg (BaseLinear.combineEvaluations PiDEC.radixWeight)
        funext child
        exact childrenMatch child
      _ = PackedBlockAction.packedYZcol covers
          (PiDEC.Raw.recomposeAssignment childAssignments) point :=
        (PackedBlockAction.PiDEC.packedYZcol_piDecRecompose
          covers childAssignments point).symm
      _ = PackedBlockAction.packedYZcol covers parentAssignment point := by
        rw [parentRecomposes]
  · exact Or.inr badRoot

/-- Conditional source-authority composition through the optional `Pi_DEC`
route. False source sidecars become either the existing mixing collision or
the delayed projection's degree-53 bad root. -/
theorem sourceBound_or_mixingCollision_or_badRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {sourceCount : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin sourceCount -> RingF)
    (sourceAssignments : Fin sourceCount ->
      PackedBlockAction.SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (sourceClaims : Fin sourceCount -> RingK)
    (parentAssignment : PackedBlockAction.SemanticAssignment shape)
    (childAssignments : Fin productionGlobalParams.k ->
      PackedBlockAction.SemanticAssignment shape)
    (combinedClaim : RingK)
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K)
    (aggregate : PiRlcSidecar.AggregateEquation challenges sourceClaims
      combinedClaim)
    (parentAssignmentMatches : PiRlcSidecar.ParentAssignmentMatches challenges
      sourceAssignments parentAssignment)
    (parentRecomposes :
      parentAssignment = PiDEC.Raw.recomposeAssignment childAssignments)
    (childrenMatch : forall child,
      childClaims child =
        PackedBlockAction.packedYZcol covers (childAssignments child) point)
    (projectionAccepted : Accepted combinedClaim childClaims producerBeta) :
    PiRlcSidecar.SourceBound covers sourceAssignments point sourceClaims ∨
      PiRlcSidecar.MixingCollision covers challenges sourceAssignments point
        sourceClaims ∨
      ProjectionCheck.BadRoot projectionOps
        (identity combinedClaim childClaims producerBeta) := by
  rcases accepted_implies_parentProjection_or_badRoot covers parentAssignment
      childAssignments point combinedClaim childClaims producerBeta
      parentRecomposes childrenMatch projectionAccepted with
    parentProjection | badRoot
  · rcases PiRlcSidecar.sourceBound_or_mixingCollision covers challenges
        sourceAssignments point sourceClaims parentAssignment combinedClaim
        aggregate parentProjection parentAssignmentMatches with
      sourceBound | mixingCollision
    · exact Or.inl sourceBound
    · exact Or.inr (Or.inl mixingCollision)
  · exact Or.inr (Or.inr badRoot)

/-- Source aggregate and parent assignment are computed, while child
recomposition remains an explicit optional premise. -/
theorem transitionAccepted_implies_sourceBound_or_mixingCollision_or_badRoot
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {sourceCount : Nat}
    (covers : domain.Covers shape)
    (challenges : Fin sourceCount -> RingF)
    (sourceAssignments : Fin sourceCount ->
      PackedBlockAction.SemanticAssignment shape)
    (point : CubePoint K domain.blockVariables)
    (sourceClaims : Fin sourceCount -> RingK)
    (childAssignments : Fin productionGlobalParams.k ->
      PackedBlockAction.SemanticAssignment shape)
    (childClaims : Fin productionGlobalParams.k -> RingK)
    (producerBeta : K)
    (parentRecomposes :
      PiRLCFinite.Raw.combineAssignments challenges sourceAssignments =
        PiDEC.Raw.recomposeAssignment childAssignments)
    (childrenMatch : forall child,
      childClaims child =
        PackedBlockAction.packedYZcol covers (childAssignments child) point)
    (projectionAccepted : TransitionAccepted challenges sourceClaims
      childClaims producerBeta) :
    PiRlcSidecar.SourceBound covers sourceAssignments point sourceClaims ∨
      PiRlcSidecar.MixingCollision covers challenges sourceAssignments point
        sourceClaims ∨
      ProjectionCheck.BadRoot projectionOps
        (identity (sourceAggregate challenges sourceClaims) childClaims
          producerBeta) := by
  exact sourceBound_or_mixingCollision_or_badRoot covers challenges
    sourceAssignments point sourceClaims
    (PiRLCFinite.Raw.combineAssignments challenges sourceAssignments)
    childAssignments (sourceAggregate challenges sourceClaims) childClaims
    producerBeta rfl rfl parentRecomposes childrenMatch projectionAccepted

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PiDecRecomposition
