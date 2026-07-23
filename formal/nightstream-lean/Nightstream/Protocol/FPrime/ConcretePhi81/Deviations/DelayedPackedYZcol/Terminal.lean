import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction

/-!
Terminal authority for the one-fold delayed packed-`y_zcol` deviation.

Assurance tier: model-level.

Owns: the minimal final-opening relation over the fourteen ordered raw child
assignments; exact commitment and fresh-norm authority for those assignments;
the complete 54-lane projection equation at the carried old block; and its
reduction to predecessor packed-`y_zcol` authority or a named binding event.

Does not own: public-input or ordinary CCS-evaluation checks, `y_ring`, child
evaluation sidecars, digest authority, concrete rows, transcript hashing,
commitment hardness, costs, or row removal.

Emits constraints: no.

Authority boundary: the projection is recomputed from the radix recomposition
of the raw assignments that open the actual ordered `Pi_DEC` children. No
child `y_zcol` value or digest supplies the result.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.terminal.openings` | bind all fourteen raw child assignments by commitment and fresh norm, then recompute their 54-lane projection | checked/security boundary | `ProjectionOpeningAccepted` |
| `fprime.delayed.terminal.close` | reduce exact final openings to predecessor packed-`y_zcol` authority or a named mixing/opening event | derived/security partition | `projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Terminal

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Minimal terminal facts over the fourteen ordered raw child assignments.
The relation retains only the obligations needed by delayed packed-`y_zcol`
authority: the actual child commitments, their fresh bound, and the exact
54-lane projection of their radix recomposition. -/
structure ProjectionOpeningAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Prop where
  childCommitment : forall child,
    commit context.key (rawChildren child) =
      (outputChildren context certificate child).commitment
  childNorm : forall child,
    Phi81Relation.assignmentNormBounded
      ((outputChildren context certificate child).stage.bound
        productionGlobalParams)
      (rawChildren child)
  projection :
    (DelayedProduction.outgoingPending context certificate).parentYZcol =
      PackedBlockAction.packedYZcol context.covers
        (PiDEC.Raw.recomposeAssignment rawChildren)
        (DelayedProduction.outgoingPending context certificate).oldBlock

/-- Exact terminal openings bind the predecessor packed output. If the raw
children do not recompose to the canonical parent opening, strict `Pi_DEC`
exhibits the standard parent-opening collision. If they do, the only
remaining algebraic branch is the existing `Pi_RLC` source-mixing event. -/
theorem projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (parentBound : DelayedRawChildren.CanonicalParentBinding context data
      certificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate))
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : ProjectionOpeningAccepted context certificate rawChildren) :
    _root_.Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        context.covers data
        (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics context.key) productionGlobalParams
        (derive context certificate).piRlcOutput.commitment) := by
  have childCommitments : forall child,
      (semantics context.key).commit (rawChildren child) =
        (((derive context certificate).piDecAttempt certificate).children
          child).commitment := by
    intro child
    simpa [outputChildren] using accepted.childCommitment child
  have childNorms : forall child,
      (semantics context.key).normBounded productionGlobalParams.b
        (rawChildren child) := by
    intro child
    simpa [outputChildren, production_norm_stages.1] using
      accepted.childNorm child
  rcases
      DelayedRawChildren.rawChildren_recompose_eq_canonicalParent_or_bindingCollision
        context data certificate piDecAccepted parentBound rawChildren
        childCommitments childNorms with
    recomposesCanonical | bindingCollision
  · rcases
      DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
        context data certificate rawChildren recomposesCanonical
        accepted.projection with packed | mixing
    · exact Or.inl (by simpa using packed)
    · exact Or.inr (Or.inl mixing)
  · exact Or.inr (Or.inr bindingCollision)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Terminal
