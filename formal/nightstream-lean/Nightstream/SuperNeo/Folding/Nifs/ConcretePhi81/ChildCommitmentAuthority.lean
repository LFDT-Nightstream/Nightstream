import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

/-!
Ordered child-commitment authority for one PiDEC family.

Assurance tier: model-level obligation reduction.

Owns: the proof that one per-step point plus the fixed-arity ordered child
commitment vector recovers every paper CE child statement when relation
structure is separately verifier-bound and both views have explicit valid
child openings, or identifies one concrete fresh-bound commitment-opening
collision. Strict PiDEC then recovers the recomposition parent as well.

Does not own: extraction of the current child openings, Ajtai/MSIS security,
Poseidon2 encoding or binding, implementation sidecars, Rust/R1CS refinement,
costs, or row removal.

Emits constraints: no.

Authority boundary: commitments alone are sufficient only after valid CE
openings bind public inputs and evaluations to assignments. Relation structure
is verifier-owned setup and remains an explicit equality premise rather than
being rehashed. The per-step point remains in the payload. The theorem never
promotes a digest or a publicly recomposed PiDEC parent into opening authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.child_commitments.structure` | both families use the same verifier-owned relation structure | computed/setup premise | `children_eq_or_freshBindingCollision` |
| `nifs.pi_dec.child_commitments.point` | bind the common per-step point once | authoritative payload | `CommitmentFamilyPayload.point` |
| `nifs.pi_dec.child_commitments.ordered` | bind every child commitment in exact type-level index order | authoritative payload | `CommitmentFamilyPayload.children` |
| `nifs.pi_dec.child_commitments.openings` | each child statement has an explicit fresh CE opening | extraction/security premise | `children_eq_or_freshBindingCollision` |
| `nifs.pi_dec.child_commitments.binding` | unequal openings under one child commitment expose a bounded collision | security boundary | `Opening.BindingCollision` |
| `nifs.pi_dec.child_commitments.exact` | equal assignments derive equal public inputs, evaluations, and complete child statements | derived | `children_eq_or_freshBindingCollision` |
| `fprime.accumulator.child_commitments.parent` | strict PiDEC derives the parent from the recovered child vector | derived | `parent_children_eq_or_freshBindingCollision` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

universe uAssignment uCommitment

/-- Candidate minimal direct paper carrier for one PiDEC family. Arity is in
the type, relation structure is verifier-owned setup, the point occurs once,
and the fresh stage is verifier-fixed. Public inputs and evaluations are
derived from the explicit openings used by the theorem. -/
structure CommitmentFamilyPayload
    (shape : Shape)
    (Commitment : Type uCommitment)
    (count : Nat) where
  point : Point shape
  children : Fin count -> Commitment

/-- Project a parent-plus-children view to the per-step point and ordered child
commitments. -/
def commitmentFamilyPayload
    {shape : Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (parent : CEStatement shape Commitment)
    (children : Fin count -> CEStatement shape Commitment) :
    CommitmentFamilyPayload shape Commitment count where
  point := parent.point
  children := fun child => (children child).commitment

private theorem payload_eq_of_fields
    {shape : Shape}
    {Commitment : Type uCommitment}
    (left right : PiDecChildPayload shape Commitment)
    (commitmentEq : left.commitment = right.commitment)
    (publicInputEq : left.publicInput = right.publicInput)
    (evaluationsEq : left.evaluations = right.evaluations) :
    left = right := by
  rcases left with ⟨leftCommitment, leftPublicInput, leftEvaluations⟩
  rcases right with ⟨rightCommitment, rightPublicInput, rightEvaluations⟩
  cases commitmentEq
  cases publicInputEq
  cases evaluationsEq
  rfl

/-- Equal points and ordered child commitments, plus separately bound relation
structure, recover the complete paper child vector from two explicit
valid-opening views, unless one child commitment has two distinct fresh-bound
openings. -/
theorem children_eq_or_freshBindingCollision
    {shape : Shape}
    {Assignment : Type uAssignment}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      (Structure shape) Assignment (PublicInput shape) (Point shape)
      Evaluation Commitment}
    {params : GlobalParams}
    {count : Nat}
    {leftParent rightParent : CEStatement shape Commitment}
    {left right : Fin count -> CEStatement shape Commitment}
    (leftAssignments rightAssignments : Fin count -> Assignment)
    (leftCanonical : CanonicalFamily leftParent left)
    (rightCanonical : CanonicalFamily rightParent right)
    (leftValid : forall child,
      CE.Holds semantics params (left child) (leftAssignments child))
    (rightValid : forall child,
      CE.Holds semantics params (right child) (rightAssignments child))
    (sameStructure :
      leftParent.constraintSystem = rightParent.constraintSystem)
    (same :
      commitmentFamilyPayload leftParent left =
        commitmentFamilyPayload rightParent right) :
    left = right ∨
      ∃ child, Nonempty
        (Opening.BindingCollision semantics params.b
          (left child).commitment) := by
  have pointEq : leftParent.point = rightParent.point :=
    congrArg (fun payload => payload.point) same
  have commitmentFunctionsEq :
      (fun child => (left child).commitment) =
        (fun child => (right child).commitment) :=
    congrArg (fun payload => payload.children) same
  have rightCanonicalForLeft : CanonicalFamily leftParent right := by
    intro child
    exact {
      relationStructure :=
        (rightCanonical child).relationStructure.trans sameStructure.symm
      point := (rightCanonical child).point.trans pointEq.symm
      stage := (rightCanonical child).stage
    }
  by_cases assignmentsEq : leftAssignments = rightAssignments
  · subst rightAssignments
    apply Or.inl
    funext child
    apply eq_of_payload_eq leftParent (left child) (right child)
        (leftCanonical child) (rightCanonicalForLeft child)
    have commitmentEq :
        (left child).commitment = (right child).commitment :=
      congrFun commitmentFunctionsEq child
    have publicInputEq :
        (left child).publicInput = (right child).publicInput := by
      calc
        (left child).publicInput =
            semantics.projectPublicInput (leftAssignments child) :=
          (leftValid child).1.2.1.symm
        _ = (right child).publicInput := (rightValid child).1.2.1
    have evaluationsEq :
        (left child).evaluations = (right child).evaluations := by
      calc
        (left child).evaluations =
            semantics.evaluations (left child).constraintSystem
              (leftAssignments child) (left child).point :=
          (leftValid child).2.2.symm
        _ = semantics.evaluations (right child).constraintSystem
              (leftAssignments child) (right child).point := by
          rw [(leftCanonical child).relationStructure,
            (rightCanonicalForLeft child).relationStructure,
            (leftCanonical child).point,
            (rightCanonicalForLeft child).point]
        _ = (right child).evaluations := (rightValid child).2.2
    exact payload_eq_of_fields _ _ commitmentEq publicInputEq evaluationsEq
  · apply Or.inr
    have differs : ∃ child,
        leftAssignments child ≠ rightAssignments child := by
      exact Classical.byContradiction fun noDifference => assignmentsEq (by
        funext child
        exact Classical.byContradiction fun different =>
          noDifference ⟨child, different⟩)
    rcases differs with ⟨child, different⟩
    refine ⟨child, ⟨{
      leftOpening := leftAssignments child
      rightOpening := rightAssignments child
      leftCommits := (leftValid child).1.1
      rightCommits := (rightValid child).1.1.trans
        (congrFun commitmentFunctionsEq child).symm
      leftNorm := ?_
      rightNorm := ?_
      different := different
    }⟩⟩
    · have bounded := (leftValid child).1.2.2
      simpa [(leftCanonical child).stage] using bounded
    · have bounded := (rightValid child).1.2.2
      simpa [(rightCanonical child).stage] using bounded

/-- Under strict PiDEC acceptance, the same reduced carrier recovers both the
exact ordered child vector and its recomposition parent, modulo the same
per-child opening-binding event. -/
theorem parent_children_eq_or_freshBindingCollision
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
    (kPositive : 0 < params.k)
    (leftAssignments rightAssignments : Fin params.k -> Assignment)
    (leftAccepted : PiDEC.Accepted algebra {
      parent := leftParent
      children := left
    })
    (rightAccepted : PiDEC.Accepted algebra {
      parent := rightParent
      children := right
    })
    (leftValid : forall child,
      CE.Holds semantics params (left child) (leftAssignments child))
    (rightValid : forall child,
      CE.Holds semantics params (right child) (rightAssignments child))
    (sameStructure :
      leftParent.constraintSystem = rightParent.constraintSystem)
    (same :
      commitmentFamilyPayload leftParent left =
        commitmentFamilyPayload rightParent right) :
    (leftParent = rightParent ∧ left = right) ∨
      ∃ child, Nonempty
        (Opening.BindingCollision semantics params.b
          (left child).commitment) := by
  rcases children_eq_or_freshBindingCollision leftAssignments rightAssignments
      (canonicalFamily_of_accepted leftAccepted)
      (canonicalFamily_of_accepted rightAccepted) leftValid rightValid
      sameStructure same with
    childrenEq | collision
  · exact Or.inl ⟨PiDEC.Accepted.parent_eq_of_children_eq kPositive
      leftAccepted rightAccepted childrenEq, childrenEq⟩
  · exact Or.inr collision

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority
