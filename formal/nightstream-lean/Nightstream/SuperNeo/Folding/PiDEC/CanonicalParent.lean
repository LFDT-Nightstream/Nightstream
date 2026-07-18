import Nightstream.SuperNeo.Folding.PiDEC

/-!
Canonical public PiDEC parent computed from a nonempty child family.

Assurance tier: model-level.

Owns: deterministic construction of every parent statement field from the
children, exact equivalence between canonical-parent acceptance and child
stage/structure/point compatibility, uniqueness against any other accepted
parent, and reconstruction of a valid parent opening from valid child
openings.

Does not own: a concrete child carrier, proof that recursive child openings
exist, injectivity from parent back to children, commitment binding, Rust,
R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: commitment, public input, and evaluations are recomputed
from all children with verifier-owned algebra. Structure and point are copied
from the first child, which is available only under `k > 0`; every other
child must match them. No digest or caller-supplied parent is authority.
Canonical construction is unique for a fixed child vector, but different
valid child vectors may recompose to the same parent.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_dec.canonical_parent.structure` | copy the first child structure and require all children to match | computed + compatibility | `parent`, `Compatible.sameStructure` |
| `pi_dec.canonical_parent.point` | copy the first child point and require all children to match | computed + compatibility | `parent`, `Compatible.samePoint` |
| `pi_dec.canonical_parent.stage` | parent is combined; children are fresh | computed + compatibility | `parent`, `Compatible.childFresh` |
| `pi_dec.canonical_parent.payload` | recompose commitment, public input, and evaluations from all children | computed | `parent` |
| `pi_dec.canonical_parent.exact` | canonical public acceptance iff child compatibility | exact model theorem | `accepted_iff_compatible` |
| `pi_dec.canonical_parent.unique` | every accepted parent over the same children equals the computed parent | derived | `eq_canonical_of_accepted` |
| `pi_dec.canonical_parent.opening` | valid child openings construct a valid computed-parent opening | derived | `holds_of_children` |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent

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

/-- Canonical first child of a nonempty PiDEC family. -/
def first (kPositive : 0 < params.k) : Fin params.k :=
  ⟨0, kPositive⟩

/-- Compute the complete public parent statement from the child family. -/
def parent
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := (children (first kPositive)).constraintSystem
  commitment := algebra.recomposeCommitment (fun child =>
    (children child).commitment)
  publicInput := algebra.recomposePublicInput (fun child =>
    (children child).publicInput)
  point := (children (first kPositive)).point
  evaluations := algebra.recomposeEvaluations (fun child =>
    (children child).evaluations)
  stage := .combined

/-- Canonical attempt with no separately supplied parent payload. -/
def attempt
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Attempt Structure PublicInput Point Evaluation Commitment params where
  parent := parent algebra kPositive children
  children := children

/-- All and only the child-side compatibility facts not fixed by canonical
parent construction. -/
structure Compatible
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop where
  childFresh : forall child, (children child).stage = .fresh
  sameStructure : forall child,
    (children child).constraintSystem =
      (children (first kPositive)).constraintSystem
  samePoint : forall child,
    (children child).point = (children (first kPositive)).point

/-- Child compatibility constructs full generic PiDEC acceptance because all
parent payload equations are true by construction. -/
theorem accepted_of_compatible
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (compatible : Compatible kPositive children) :
    Accepted algebra (attempt algebra kPositive children) := by
  exact {
    parentCombined := rfl
    childFresh := compatible.childFresh
    sameStructure := compatible.sameStructure
    samePoint := compatible.samePoint
    commitmentEquation := rfl
    publicInputEquation := rfl
    evaluationEquation := rfl
  }

/-- Full acceptance of the canonical attempt exposes exactly the three child
compatibility families. -/
theorem compatible_of_accepted
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (accepted : Accepted algebra (attempt algebra kPositive children)) :
    Compatible kPositive children := by
  refine {
    childFresh := accepted.childFresh
    sameStructure := ?_
    samePoint := ?_
  }
  · intro child
    exact accepted.sameStructure child
  · intro child
    exact accepted.samePoint child

/-- Canonical PiDEC acceptance is exactly child compatibility; parent payload
comparison is absent because those fields are computed. -/
theorem accepted_iff_compatible
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Accepted algebra (attempt algebra kPositive children) <->
      Compatible kPositive children := by
  exact ⟨compatible_of_accepted algebra kPositive children,
    accepted_of_compatible algebra kPositive children⟩

/-- Any separately supplied parent accepted over the same child family equals
the canonical computed parent statement. -/
theorem eq_canonical_of_accepted
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (candidate : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (candidateAccepted : Accepted algebra {
      parent := candidate
      children := children
    }) :
    candidate = parent algebra kPositive children := by
  have compatible : Compatible kPositive children := {
    childFresh := candidateAccepted.childFresh
    sameStructure := fun child =>
      (candidateAccepted.sameStructure child).trans
        (candidateAccepted.sameStructure (first kPositive)).symm
    samePoint := fun child =>
      (candidateAccepted.samePoint child).trans
        (candidateAccepted.samePoint (first kPositive)).symm
  }
  exact Accepted.parent_eq_of_children_eq kPositive candidateAccepted
    (accepted_of_compatible algebra kPositive children compatible) rfl

/-- Valid child openings construct a valid opening of the computed parent.
This is the semantic authority needed by recursive induction; no carried
parent opening is assumed. -/
theorem holds_of_children
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (kPositive : 0 < params.k)
    (children : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (compatible : Compatible kPositive children)
    (assignments : Fin params.k -> Assignment)
    (childrenValid : forall child,
      CE.Holds semantics params (children child) (assignments child)) :
    CE.Holds semantics params (parent algebra kPositive children)
      (algebra.recomposeAssignment assignments) := by
  exact reduce_knowledge semantics params algebra
    (attempt algebra kPositive children) assignments kPositive
    (accepted_of_compatible algebra kPositive children compatible)
    childrenValid

end

end Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent
