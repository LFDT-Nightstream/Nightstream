import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority

/-!
Canonical-parent authority for one PiDEC family.

Assurance tier: model-level conditional obligation reduction.

Owns: the proof that, when each child family is the deterministic radix split
of an explicit valid combined parent opening, one per-step point plus one parent
commitment recovers the exact parent and ordered children, or exposes two
distinct `B`-bounded openings of that parent commitment.

Does not own: extraction or verification of canonical child openings, concrete
Ajtai/MSIS security, Poseidon2 encoding or binding, implementation sidecars,
Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: this reduction is strictly stronger than public PiDEC
acceptance. Public recomposition permits signed-digit child substitutions.
The smaller carrier is sound only under `PiDEC.CanonicalChildren.ForOpening`
for both views and separately verifier-bound relation structure.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.canonical_parent.structure` | both parents use verifier-owned relation structure | computed/setup premise | `parent_opening_eq_or_bindingCollision` |
| `nifs.pi_dec.canonical_parent.point` | bind the per-step evaluation point once | authoritative payload | `CanonicalParentPayload.point` |
| `nifs.pi_dec.canonical_parent.commitment` | bind one combined parent commitment | authoritative payload | `CanonicalParentPayload.commitment` |
| `nifs.pi_dec.canonical_parent.opening` | each parent has an explicit valid combined opening | extraction/security premise | `PiDEC.CanonicalChildren.ForOpening` |
| `nifs.pi_dec.canonical_parent.children` | every child is the deterministic split of that opening | canonicality premise | `PiDEC.CanonicalChildren.ForOpening.childrenEq` |
| `nifs.pi_dec.canonical_parent.binding` | unequal parent openings expose one `B`-bounded collision | security boundary | `Opening.BindingCollision` |
| `fprime.accumulator.canonical_parent.exact` | equal compact carriers recover exact parent and children | derived | `parent_children_eq_or_bindingCollision` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding

universe uAssignment uCommitment

/-- Smallest direct paper carrier currently justified for a canonically bound
PiDEC family. Relation structure is setup; combined stage is verifier-fixed. -/
structure CanonicalParentPayload
    (shape : Shape)
    (Commitment : Type uCommitment) where
  point : Point shape
  commitment : Commitment

/-- Project one combined parent to its per-step point and commitment. -/
def canonicalParentPayload
    {shape : Shape}
    {Commitment : Type uCommitment}
    (parent : CEStatement shape Commitment) :
    CanonicalParentPayload shape Commitment where
  point := parent.point
  commitment := parent.commitment

private theorem statement_eq_of_fields
    {shape : Shape}
    {Commitment : Type uCommitment}
    (left right : CEStatement shape Commitment)
    (structureEq : left.constraintSystem = right.constraintSystem)
    (commitmentEq : left.commitment = right.commitment)
    (publicInputEq : left.publicInput = right.publicInput)
    (pointEq : left.point = right.point)
    (evaluationsEq : left.evaluations = right.evaluations)
    (stageEq : left.stage = right.stage) :
    left = right := by
  rcases left with
    ⟨leftStructure, leftCommitment, leftPublicInput, leftPoint,
      leftEvaluations, leftStage⟩
  rcases right with
    ⟨rightStructure, rightCommitment, rightPublicInput, rightPoint,
      rightEvaluations, rightStage⟩
  cases structureEq
  cases commitmentEq
  cases publicInputEq
  cases pointEq
  cases evaluationsEq
  cases stageEq
  rfl

/-- Equal point-plus-parent-commitment carriers recover both the complete
parent statement and its opening, unless the commitment has two distinct
valid combined openings. -/
theorem parent_opening_eq_or_bindingCollision
    {shape : Shape}
    {Assignment : Type uAssignment}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      (Structure shape) Assignment (PublicInput shape) (Point shape)
      Evaluation Commitment}
    {params : GlobalParams}
    {leftParent rightParent : CEStatement shape Commitment}
    (leftAssignment rightAssignment : Assignment)
    (leftCombined : leftParent.stage = .combined)
    (rightCombined : rightParent.stage = .combined)
    (leftValid : CE.Holds semantics params leftParent leftAssignment)
    (rightValid : CE.Holds semantics params rightParent rightAssignment)
    (sameStructure :
      leftParent.constraintSystem = rightParent.constraintSystem)
    (same :
      canonicalParentPayload leftParent = canonicalParentPayload rightParent) :
    (leftParent = rightParent ∧ leftAssignment = rightAssignment) ∨
      Nonempty
        (Opening.BindingCollision semantics params.bigB
          leftParent.commitment) := by
  have pointEq : leftParent.point = rightParent.point :=
    congrArg (fun payload => payload.point) same
  have commitmentEq : leftParent.commitment = rightParent.commitment :=
    congrArg (fun payload => payload.commitment) same
  by_cases assignmentEq : leftAssignment = rightAssignment
  · subst rightAssignment
    have publicInputEq :
        leftParent.publicInput = rightParent.publicInput := by
      calc
        leftParent.publicInput =
            semantics.projectPublicInput leftAssignment :=
          leftValid.1.2.1.symm
        _ = rightParent.publicInput := rightValid.1.2.1
    have evaluationsEq :
        leftParent.evaluations = rightParent.evaluations := by
      calc
        leftParent.evaluations =
            semantics.evaluations leftParent.constraintSystem leftAssignment
              leftParent.point := leftValid.2.2.symm
        _ = semantics.evaluations rightParent.constraintSystem leftAssignment
              rightParent.point := by rw [sameStructure, pointEq]
        _ = rightParent.evaluations := rightValid.2.2
    have stageEq : leftParent.stage = rightParent.stage :=
      leftCombined.trans rightCombined.symm
    exact Or.inl ⟨statement_eq_of_fields leftParent rightParent
      sameStructure commitmentEq publicInputEq pointEq evaluationsEq stageEq,
      rfl⟩
  · apply Or.inr
    exact ⟨{
      leftOpening := leftAssignment
      rightOpening := rightAssignment
      leftCommits := leftValid.1.1
      rightCommits := rightValid.1.1.trans commitmentEq.symm
      leftNorm := by
        simpa [leftCombined] using leftValid.1.2.2
      rightNorm := by
        simpa [rightCombined] using rightValid.1.2.2
      different := assignmentEq
    }⟩

/-- Canonical child authority collapses the complete PiDEC family to one
parent commitment and point, modulo the parent-opening collision above. -/
theorem parent_children_eq_or_bindingCollision
    {shape : Shape}
    {Assignment : Type uAssignment}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      (Structure shape) Assignment (PublicInput shape) (Point shape)
      Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Structure shape) Assignment (PublicInput shape) (Point shape)
      Evaluation Commitment semantics params}
    {leftParent rightParent : CEStatement shape Commitment}
    {left right : Fin params.k -> CEStatement shape Commitment}
    {leftAssignment rightAssignment : Assignment}
    (leftCanonical : PiDEC.CanonicalChildren.ForOpening algebra leftParent
      leftAssignment left)
    (rightCanonical : PiDEC.CanonicalChildren.ForOpening algebra rightParent
      rightAssignment right)
    (sameStructure :
      leftParent.constraintSystem = rightParent.constraintSystem)
    (same :
      canonicalParentPayload leftParent = canonicalParentPayload rightParent) :
    (leftParent = rightParent ∧ left = right) ∨
      Nonempty
        (Opening.BindingCollision semantics params.bigB
          leftParent.commitment) := by
  rcases parent_opening_eq_or_bindingCollision leftAssignment rightAssignment
      leftCanonical.parentCombined rightCanonical.parentCombined
      leftCanonical.parentValid rightCanonical.parentValid sameStructure same with
    exactOpening | collision
  · rcases exactOpening with ⟨parentEq, assignmentEq⟩
    subst rightParent
    subst rightAssignment
    exact Or.inl ⟨rfl, leftCanonical.children_eq rightCanonical⟩
  · exact Or.inr collision

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority
