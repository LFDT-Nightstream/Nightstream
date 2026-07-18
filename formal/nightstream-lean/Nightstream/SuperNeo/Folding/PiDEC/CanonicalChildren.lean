import Nightstream.SuperNeo.Folding.PiDEC

/-!
Canonical PiDEC children bound to one valid parent opening.

Assurance tier: model-level.

Owns: the exact semantic predicate that a public child family is the
verifier-owned radix split of one valid combined parent opening; perfect
PiDEC completeness for that predicate; uniqueness for a fixed opening; and
the explicit parent-opening binding failure separating two candidate splits.

Does not own: extraction of a parent opening, computational commitment
binding, a concrete child carrier, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: public PiDEC recomposition is necessary but not sufficient
for this predicate. The child family is authoritative only after it is tied to
the same parent opening, or after distinct parent openings are reduced to the
explicit binding-collision event below.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.canonical_children.parent` | one combined parent has a valid opening | semantic input | `ForOpening.parentCombined`, `ForOpening.parentValid` |
| `nifs.pi_dec.canonical_children.split` | children equal the deterministic split of that opening | computed equality | `ForOpening.childrenEq` |
| `nifs.pi_dec.canonical_children.acceptance` | canonical children satisfy strict public PiDEC and CE membership | derived | `ForOpening.complete` |
| `nifs.pi_dec.canonical_children.unique` | one fixed opening determines one child family | derived | `ForOpening.children_eq` |
| `nifs.pi_dec.canonical_children.binding` | two existential splits agree or expose a parent-opening collision | security boundary | `children_eq_or_bindingCollision` |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

section

variable {Structure : Type uStructure}
variable {Assignment : Type uAssignment}
variable {PublicInput : Type uPublicInput}
variable {Point : Type uPoint}
variable {Evaluation : Type uEvaluation}
variable {Commitment : Type uCommitment}
variable {semantics : RelationSemantics
  Structure Assignment PublicInput Point Evaluation Commitment}
variable {params : GlobalParams}

/-- Exact child authority relative to one explicit parent opening. -/
structure ForOpening
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop where
  parentCombined : parent.stage = .combined
  parentValid : CE.Holds semantics params parent assignment
  childrenEq : children = childrenOf algebra parent assignment

namespace ForOpening

/-- Canonical child authority implies both strict public PiDEC acceptance and
the exact private opening of every child. -/
theorem complete
    {algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {assignment : Assignment}
    {children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (bound : ForOpening algebra parent assignment children) :
    Accepted algebra { parent := parent, children := children } /\
      forall child,
        CE.Holds semantics params (children child)
          (algebra.splitAssignment assignment child) := by
  rw [bound.childrenEq]
  exact PiDEC.complete semantics params algebra parent assignment
    bound.parentCombined bound.parentValid

/-- One fixed parent opening has one deterministic canonical child family. -/
theorem children_eq
    {algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {assignment : Assignment}
    {left right : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (leftBound : ForOpening algebra parent assignment left)
    (rightBound : ForOpening algebra parent assignment right) :
    left = right :=
  leftBound.childrenEq.trans rightBound.childrenEq.symm

end ForOpening

/-- Canonical child authority when the parent opening remains existential. -/
def Holds
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop :=
  exists assignment, ForOpening algebra parent assignment children

/-- Two child families canonically derived from one public parent are equal,
unless that parent commitment has two distinct valid combined openings. -/
theorem children_eq_or_bindingCollision
    {algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {left right : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (leftBound : Holds algebra parent left)
    (rightBound : Holds algebra parent right) :
    left = right ∨
      Nonempty (ParentOpeningBindingCollision semantics params
        parent.commitment) := by
  rcases leftBound with ⟨leftAssignment, leftBound⟩
  rcases rightBound with ⟨rightAssignment, rightBound⟩
  by_cases same : leftAssignment = rightAssignment
  · subst rightAssignment
    exact Or.inl (leftBound.children_eq rightBound)
  · apply Or.inr
    exact ⟨{
      parentOpening := leftAssignment
      recomposedOpening := rightAssignment
      parentCommits := leftBound.parentValid.1.1
      recomposedCommits := rightBound.parentValid.1.1
      parentNorm := by
        simpa [leftBound.parentCombined] using leftBound.parentValid.1.2.2
      recomposedNorm := by
        simpa [rightBound.parentCombined] using rightBound.parentValid.1.2.2
      different := same
    }⟩

/-- Outside the explicit parent-opening collision event, an existentially
canonical child family is unique. -/
theorem children_eq_of_no_bindingCollision
    {algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {left right : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (noCollision :
      ¬ Nonempty (ParentOpeningBindingCollision semantics params
        parent.commitment))
    (leftBound : Holds algebra parent left)
    (rightBound : Holds algebra parent right) :
    left = right := by
  rcases children_eq_or_bindingCollision leftBound rightBound with
    equal | collision
  · exact equal
  · exact False.elim (noCollision collision)

end

end Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren
