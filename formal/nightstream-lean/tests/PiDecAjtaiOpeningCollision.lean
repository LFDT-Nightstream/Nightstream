import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecAjtaiOpeningCollision

/-! Narrow compile-time checks for the model-level Π_DEC-to-concrete-Ajtai
collision specialization. -/

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.PiDecAjtaiOpeningCollision

example
    {context : Context}
    {params : GlobalParams}
    {commitment : Commitment}
    {assignmentWidth : Nat}
    (collision : PiDEC.ParentOpeningBindingCollision
      (relationSemantics context) params commitment)
    (parentWidth : collision.parentOpening.length = assignmentWidth)
    (recomposedWidth : collision.recomposedOpening.length = assignmentWidth) :
    Nonempty (AjtaiOpeningCollision context params commitment assignmentWidth) :=
  parentOpeningBindingCollision_to_ajtaiOpeningCollision
    collision parentWidth recomposedWidth

example
    {context : Context}
    {params : GlobalParams}
    {commitment : Commitment}
    {assignmentWidth : Nat}
    (collision : PiDEC.ParentOpeningBindingCollision
      (relationSemantics context) params commitment)
    (parentWidth : collision.parentOpening.length = assignmentWidth)
    (recomposedWidth : collision.recomposedOpening.length = assignmentWidth) :
    ∃ concrete : AjtaiOpeningCollision
        context params commitment assignmentWidth,
      packAssignment concrete.opening1 ≠
        packAssignment concrete.opening2 := by
  rcases parentOpeningBindingCollision_to_ajtaiOpeningCollision
    collision parentWidth recomposedWidth with ⟨concrete⟩
  exact ⟨concrete, concrete.packedDistinct⟩
