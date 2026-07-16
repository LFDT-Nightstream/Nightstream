import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear

/-!
Parent-opening authority bridge for the combined `Pi_RLC` `yZcol` value.

Protocol: SuperNeo `Pi_RLC` combined parent followed by `Pi_DEC`.
Phase: accepted parent/child CE openings to combined-parent recomposition.
Constraint family: parent commitment binding; this file emits no rows.

Owns: specialization of the generic `PiDEC` parent-opening dichotomy to any
ring-aligned typed Phi81 public carrier; and transport of all 54 `yZcol` lanes
of one independently valid combined parent, or a named binding collision.

Does not own: specialization to the proposed five-ring-column/270-field public
carrier, implementation of the thirteen fixed padding coordinates, construction
of the complete Phi81 `PiDEC.Algebra`, Ajtai/MSIS hardness, elimination of the
collision branch, proof that NIFS acceptance supplies these CE openings,
transcript derivation or equality of `rPrime`/`sPrime`, the preceding `PiRLC`
RingF action that combines the `PiCCS` source product into this one parent,
output-message authority, Rust/R1CS conformance, row removal, or constraint
counts.

Emits constraints: no.

Authority boundary: the single combined parent and all child assignments are
independently required to satisfy `CE.Holds`. `PiDEC.Accepted` contributes only
its public recomposition equations. The parent assignment equals the concrete
production recomposition only outside `ParentOpeningBindingCollision`. No
claim is made here that the parent is the `PiRLC` action on `PiCCS` sources. The
separate
`UsesProductionAssignmentRecomposition` premise records the currently missing
complete parent-assignment recomposition refinement; it is not inferred from a
digest.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.authority.parent_opening.public_shape` | public width is a verifier-owned whole number of Phi81 rings and fits the complete carrier | typed input | `ConcretePiDEC.RelationShape` |
| `nifs.pi_dec.verify.authority.parent_opening.ce` | the combined parent and every radix child have independently valid CE openings | checked premise | `CE.Holds` inputs |
| `nifs.pi_dec.verify.authority.parent_opening.accepted` | public parent fields are the verifier-checked recomposition of child fields | checked premise | `PiDEC.Accepted` input |
| `nifs.pi_dec.verify.authority.parent_opening.recomposition` | the abstract algebra uses the canonical production assignment recomposition | refinement premise | `UsesProductionAssignmentRecomposition` |
| `nifs.pi_dec.verify.authority.parent_opening.binding` | parent assignment equals child recomposition or exposes two distinct `B`-bounded openings | derived | `parentAssignment_eq_recompose_or_bindingCollision` |
| `nifs.pi_dec.verify.authority.parent_opening.y_zcol` | all 54 combined-parent lanes transport, or the same parent exposes the collision | derived | `parentYZcol_transport_or_bindingCollision` |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol` | source-product `yZcol` claims determine this combined parent | missing upstream bridge | not owned here |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

universe uCommitment

namespace ConcretePiDEC

/-- A paper-valid public carrier has a verifier-owned whole-ring width. The
current 257-field implementation is deliberately not representable here. -/
abbrev RelationShape
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth) :
    Phi81Relation.Shape :=
  Phi81Relation.Shape.ofSemantic shape publicRingColumns publicFits

abbrev Semantics
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth)
    (Commitment : Type uCommitment)
    (commit : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) -> Commitment) :=
  Phi81Relation.relationSemantics commit

abbrev Algebra
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth)
    (Commitment : Type uCommitment)
    (commit : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) -> Commitment) :=
  PiDEC.Algebra
    (Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.Point
      (RelationShape shape publicRingColumns publicFits))
    Phi81Relation.Evaluation Commitment
    (Semantics shape publicRingColumns publicFits Commitment commit)
    productionGlobalParams

abbrev Attempt
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth)
    (Commitment : Type uCommitment) :=
  PiDEC.Attempt
    (Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.Point
      (RelationShape shape publicRingColumns publicFits))
    Phi81Relation.Evaluation Commitment productionGlobalParams

/-- The one field of the not-yet-completed concrete Phi81 `PiDEC.Algebra`
needed by this bridge: its assignment recomposition is exactly the independent
production `b = 2`, `k = 14` definition. -/
def UsesProductionAssignmentRecomposition
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    {Commitment : Type uCommitment}
    {commit : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) -> Commitment}
    (algebra : Algebra shape publicRingColumns publicFits Commitment commit) :
    Prop :=
  forall children,
    algebra.recomposeAssignment children =
      Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment children

/-- Forget only the public-input width when feeding a complete assignment to
the independently defined `yZcol` projection. Both shapes have exactly the
same logical width and therefore the same complete Phi81 carrier. -/
def toYZcolAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (assignment : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits)) :
    Phi81Relation.Assignment (BaseLinear.relationShape shape) :=
  fun column => assignment column

private theorem toYZcolAssignment_combine
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    {count : Nat}
    (weights : Fin count -> F)
    (assignments : Fin count ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    toYZcolAssignment
        (Phi81Relation.EvaluationHomomorphism.BaseLinear.combineAssignments
          weights assignments) =
      Phi81Relation.EvaluationHomomorphism.BaseLinear.combineAssignments
        weights (fun index => toYZcolAssignment (assignments index)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      funext column
      simp only [Phi81Relation.EvaluationHomomorphism.BaseLinear.combineAssignments,
        Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentAdd,
        Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentScale,
        toYZcolAssignment]
      apply congrArg (fun tail => weights 0 * assignments 0 column + tail)
      simpa only [toYZcolAssignment] using congrFun (inductionHypothesis
          (fun index => weights index.succ)
          (fun index => assignments index.succ)) column

/-- Forgetting the public projection commutes with the exact production radix
recomposition on the complete assignment. -/
theorem toYZcolAssignment_recompose
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    toYZcolAssignment
        (Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
          children) =
      Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
        (fun index => toYZcolAssignment (children index)) := by
  unfold Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment
  exact toYZcolAssignment_combine
    Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight children

end ConcretePiDEC

/-- Independently valid parent/child CE openings plus accepted `PiDEC` checks
bind the combined parent assignment to the production recomposition, except
for the generic parent-opening binding collision. -/
theorem parentAssignment_eq_recompose_or_bindingCollision
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    {Commitment : Type uCommitment}
    (commit : Phi81Relation.Assignment
      (ConcretePiDEC.RelationShape shape publicRingColumns publicFits) ->
        Commitment)
    (algebra : ConcretePiDEC.Algebra shape publicRingColumns publicFits
      Commitment commit)
    (attempt : ConcretePiDEC.Attempt shape publicRingColumns publicFits
      Commitment)
    (parentAssignment :
      Phi81Relation.Assignment
        (ConcretePiDEC.RelationShape shape publicRingColumns publicFits))
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (ConcretePiDEC.RelationShape shape publicRingColumns publicFits))
    (usesProduction :
      ConcretePiDEC.UsesProductionAssignmentRecomposition algebra)
    (accepted : PiDEC.Accepted algebra attempt)
    (parentValid : CE.Holds
      (ConcretePiDEC.Semantics shape publicRingColumns publicFits Commitment
        commit) productionGlobalParams attempt.parent parentAssignment)
    (childrenValid : forall index,
      CE.Holds (ConcretePiDEC.Semantics shape publicRingColumns publicFits
        Commitment commit)
        productionGlobalParams (attempt.children index) (children index)) :
    parentAssignment =
        Phi81Relation.EvaluationHomomorphism.PiDEC.recomposeAssignment children ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (ConcretePiDEC.Semantics shape publicRingColumns publicFits Commitment
          commit)
        productionGlobalParams attempt.parent.commitment) := by
  rcases PiDEC.accepted_parent_eq_recompose_or_bindingCollision
      (ConcretePiDEC.Semantics shape publicRingColumns publicFits Commitment
        commit)
      productionGlobalParams algebra attempt parentAssignment children accepted
      parentValid childrenValid with equal | collision
  · exact Or.inl (equal.trans (usesProduction children))
  · exact Or.inr collision

/-- The single combined parent's independently recomputed 54-lane sidecar
transports through production `PiDEC`, or the valid parent and reconstructed
parent form the named binding collision. This theorem does not claim that the
parent is the preceding `PiRLC` combination of `PiCCS` sources. -/
theorem parentYZcol_transport_or_bindingCollision
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {publicRingColumns : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    {Commitment : Type uCommitment}
    (commit : Phi81Relation.Assignment
      (ConcretePiDEC.RelationShape shape publicRingColumns publicFits) ->
        Commitment)
    (covers : domain.Covers shape)
    (sPrime : CubePoint K domain.columnVariables)
    (algebra : ConcretePiDEC.Algebra shape publicRingColumns publicFits
      Commitment commit)
    (attempt : ConcretePiDEC.Attempt shape publicRingColumns publicFits
      Commitment)
    (parentAssignment :
      Phi81Relation.Assignment
        (ConcretePiDEC.RelationShape shape publicRingColumns publicFits))
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (ConcretePiDEC.RelationShape shape publicRingColumns publicFits))
    (usesProduction :
      ConcretePiDEC.UsesProductionAssignmentRecomposition algebra)
    (accepted : PiDEC.Accepted algebra attempt)
    (parentValid : CE.Holds
      (ConcretePiDEC.Semantics shape publicRingColumns publicFits Commitment
        commit) productionGlobalParams attempt.parent parentAssignment)
    (childrenValid : forall index,
      CE.Holds (ConcretePiDEC.Semantics shape publicRingColumns publicFits
        Commitment commit)
        productionGlobalParams (attempt.children index) (children index)) :
    BaseLinear.yZcolEvaluation covers
        (ConcretePiDEC.toYZcolAssignment parentAssignment) sPrime =
        Phi81Relation.EvaluationHomomorphism.BaseLinear.combineEvaluations
          Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
          (fun index =>
            BaseLinear.yZcolEvaluation covers
              (ConcretePiDEC.toYZcolAssignment (children index)) sPrime) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (ConcretePiDEC.Semantics shape publicRingColumns publicFits Commitment
          commit)
        productionGlobalParams attempt.parent.commitment) := by
  rcases parentAssignment_eq_recompose_or_bindingCollision commit algebra attempt
      parentAssignment children usesProduction accepted parentValid
      childrenValid with equal | collision
  · have projectedEqual := congrArg ConcretePiDEC.toYZcolAssignment equal
    rw [projectedEqual, ConcretePiDEC.toYZcolAssignment_recompose]
    exact Or.inl (BaseLinear.yZcolEvaluation_piDecRecompose covers
      (fun index => ConcretePiDEC.toYZcolAssignment (children index)) sPrime)
  · exact Or.inr collision

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening
