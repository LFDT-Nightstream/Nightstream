import Mathlib.Logic.Equiv.Defs
import NightstreamFPrime.Spec.Folding.PiDEC

/-! Exact conversion of the PiDEC parent-opening event to the generic bounded
binding event. Both openings, both equations and both norm bounds are retained.
This conversion supplies no hardness or probability assumption. -/

namespace NightstreamFPrime.Spec.Folding.PiDEC

universe uS uW uX uR uY uC

variable {Structure : Type uS} {Assignment : Type uW}
  {PublicInput : Type uX} {Point : Type uR}
  {Evaluation : Type uY} {Commitment : Type uC}
  (semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment)
  (params : GlobalParams) (commitment : Commitment)

/-- The two event records differ only in their field names. -/
def parentCollisionEquiv :
    ParentOpeningBindingCollision semantics params commitment ≃
      Opening.BindingCollision semantics params.bigB commitment where
  toFun collision := {
    leftOpening := collision.parentOpening
    rightOpening := collision.recomposedOpening
    leftCommits := collision.parentCommits
    rightCommits := collision.recomposedCommits
    leftNorm := collision.parentNorm
    rightNorm := collision.recomposedNorm
    different := collision.different }
  invFun collision := {
    parentOpening := collision.leftOpening
    recomposedOpening := collision.rightOpening
    parentCommits := collision.leftCommits
    recomposedCommits := collision.rightCommits
    parentNorm := collision.leftNorm
    recomposedNorm := collision.rightNorm
    different := collision.different }
  left_inv collision := by cases collision; rfl
  right_inv collision := by cases collision; rfl

theorem parent_bindingCollision_iff :
    Nonempty (ParentOpeningBindingCollision semantics params commitment) ↔
      Nonempty (Opening.BindingCollision semantics params.bigB commitment) := by
  constructor
  · rintro ⟨collision⟩
    exact ⟨parentCollisionEquiv semantics params commitment collision⟩
  · rintro ⟨collision⟩
    exact ⟨(parentCollisionEquiv semantics params commitment).symm collision⟩

end NightstreamFPrime.Spec.Folding.PiDEC
