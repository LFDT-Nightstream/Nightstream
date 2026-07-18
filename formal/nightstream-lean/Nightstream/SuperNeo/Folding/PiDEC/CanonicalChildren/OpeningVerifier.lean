import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren

/-!
Minimal semantic verifier for a canonical PiDEC parent opening.

Assurance tier: model-level obligation exactness.

Owns: the point-plus-parent-commitment input; deterministic construction of
the combined parent statement and its radix-split children from one private
opening; the three generic checks needed for parent CE membership; exact
soundness and completeness; and derivation of strict PiDEC acceptance and
canonical child openings.

Does not own: a concrete relation, point decoding, commitment binding,
Poseidon2, Fiat--Shamir, extraction of the opening, Rust/R1CS refinement,
costs, or row removal.

Emits constraints: no.

Authority boundary: the prover supplies only the private assignment. The
relation structure and parameters are verifier-owned; the public input,
evaluation array, combined stage, and complete child family are computed.
Consequently they are not independent equality checks in this verifier.
Computing them may still cost constraints after concrete lowering.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.canonical_opening.input` | bind one point and parent commitment | authoritative input | `Input` |
| `nifs.pi_dec.canonical_opening.commitment` | opening commits to the carried parent commitment | checked | `Accepted.commitment` |
| `nifs.pi_dec.canonical_opening.norm` | opening is bounded by verifier-owned `B = b^k` | checked | `Accepted.combinedNorm` |
| `nifs.pi_dec.canonical_opening.point` | point belongs to the verifier-owned CE domain | checked or type-derived | `Accepted.pointValid` |
| `nifs.pi_dec.canonical_opening.parent` | public input, evaluations, and stage are computed | computed | `parent` |
| `nifs.pi_dec.canonical_opening.children` | all children are the deterministic radix split | computed | `children` |
| `nifs.pi_dec.canonical_opening.exact` | accepted iff the computed parent has a valid CE opening | derived | `accepted_iff_parentHolds` |
| `nifs.pi_dec.canonical_opening.pi_dec` | canonical children satisfy strict PiDEC and fresh CE membership | derived | `piDecAccepted_and_childHolds` |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- The complete public carrier for a canonical parent opening. Relation
structure, parameters, and the combined norm stage are verifier-owned. -/
structure Input (Point : Type uPoint) (Commitment : Type uCommitment) where
  point : Point
  commitment : Commitment

/-- Construct the complete combined CE statement from verifier-owned
structure, the compact public carrier, and one private assignment. Public
input and evaluations are definitions, not caller fields. -/
def parent
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := system
  commitment := input.commitment
  publicInput := semantics.projectPublicInput assignment
  point := input.point
  evaluations := semantics.evaluations system assignment input.point
  stage := .combined

/-- The three generic semantic checks not discharged by construction. A
concrete typed relation may prove some of these intrinsic. -/
structure Accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment) : Prop where
  commitment : semantics.commit assignment = input.commitment
  combinedNorm : semantics.normBounded params.bigB assignment
  pointValid : semantics.evaluationPointValid system input.point

/-- The generic three-check verifier is sound for the independently defined
CE relation because every other parent field is computed. -/
theorem parentHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {system : Structure}
    {input : Input Point Commitment}
    {assignment : Assignment}
    (accepted : Accepted semantics params system input assignment) :
    CE.Holds semantics params
      (parent semantics system input assignment) assignment := by
  exact ⟨⟨accepted.commitment, rfl, accepted.combinedNorm⟩,
    accepted.pointValid, rfl⟩

/-- Completeness: CE membership of the deterministically materialized parent
contains exactly the three generic checks above. -/
theorem accepted_of_parentHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {system : Structure}
    {input : Input Point Commitment}
    {assignment : Assignment}
    (holds : CE.Holds semantics params
      (parent semantics system input assignment) assignment) :
    Accepted semantics params system input assignment := by
  exact {
    commitment := holds.1.1
    combinedNorm := holds.1.2.2
    pointValid := holds.2.1
  }

/-- Exactness of the independent minimal opening contract. -/
theorem accepted_iff_parentHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment) :
    Accepted semantics params system input assignment ↔
      CE.Holds semantics params
        (parent semantics system input assignment) assignment := by
  exact ⟨parentHolds, accepted_of_parentHolds⟩

/-- Deterministically materialize the complete ordered child family from the
same parent opening. -/
def children
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment
      semantics params)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment) :
    Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment :=
  PiDEC.childrenOf algebra (parent semantics system input assignment)
    assignment

/-- Acceptance supplies the exact canonical-child predicate required by the
smaller parent-only carrier. -/
theorem canonicalChildren
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment
      semantics params)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment)
    (accepted : Accepted semantics params system input assignment) :
    CanonicalChildren.ForOpening algebra
      (parent semantics system input assignment) assignment
      (children algebra system input assignment) := {
  parentCombined := rfl
  parentValid := parentHolds accepted
  childrenEq := rfl
}

/-- No separate public recomposition check is needed for computed children:
strict PiDEC acceptance and every fresh child opening follow by completeness. -/
theorem piDecAccepted_and_childHolds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra
      Structure Assignment PublicInput Point Evaluation Commitment
      semantics params)
    (system : Structure)
    (input : Input Point Commitment)
    (assignment : Assignment)
    (accepted : Accepted semantics params system input assignment) :
    PiDEC.Accepted algebra {
      parent := parent semantics system input assignment
      children := children algebra system input assignment
    } ∧
      ∀ child, CE.Holds semantics params
        (children algebra system input assignment child)
        (algebra.splitAssignment assignment child) :=
  (canonicalChildren algebra system input assignment accepted).complete

end Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier
