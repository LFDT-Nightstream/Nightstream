import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren

/-!
Minimal public authority for one canonical PiDEC child family.

Assurance tier: model-level obligation reduction.

Owns: the proof that, for children inheriting one computed parent's relation
structure and point plus the verifier-fixed fresh stage, the three-field
`PiDecChildPayload` is an injective representation of the complete paper CE
statement; and the exact shared-context-plus-children family carrier.

Does not own: private child openings, concrete claim/field serialization,
Poseidon2 parameters or security, implementation sidecars, Rust/R1CS
refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: commitment, public input, and evaluations are the only
child-specific paper fields. Relation structure, point, and stage may be
omitted from the accumulator message only when `CanonicalFor` is established
by verifier construction. Across different parent contexts, structure and
point must still be bound once per family; `FamilyPayload` does exactly that.
Rust-only sidecars require their own derivation or validation theorem; this
module does not silently authenticate them.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.child.structure` | child structure is inherited from the computed parent | computed premise | `CanonicalFor.relationStructure` |
| `nifs.pi_dec.child.point` | child point is inherited from the computed parent | computed premise | `CanonicalFor.point` |
| `nifs.pi_dec.child.stage` | child stage is verifier-fixed fresh | computed premise | `CanonicalFor.stage` |
| `nifs.pi_dec.child.commitment` | retain the complete child commitment | authoritative payload | `PiDecChildPayload.commitment` |
| `nifs.pi_dec.child.public_input` | retain all 270 public-input coordinates | authoritative payload | `PiDecChildPayload.publicInput` |
| `nifs.pi_dec.child.evaluations` | retain array length, matrix order, and every evaluation lane | authoritative payload | `PiDecChildPayload.evaluations` |
| `nifs.pi_dec.child.canonical_family` | verifier construction establishes inherited fields for every child | computed | `canonicalFamily_childrenOf`, `canonicalFamily_of_forOpening` |
| `nifs.pi_dec.child.payload_exact` | payload plus inherited fields reconstructs the exact CE statement | derived | `materialize_ofStatement`, `family_eq_of_payloadList_eq` |
| `fprime.accumulator.family.context` | bind common structure and point once rather than once per child | authoritative shared payload | `FamilyPayload`, `familyPayload` |
| `fprime.accumulator.family.exact` | shared context plus ordered child payloads recover parent and children under strict PiDEC | derived | `parent_children_eq_of_familyPayload_eq` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

universe uAssignment uCommitment

/-- The three inherited CE fields required before the reduced child payload is
an exact representation. -/
structure CanonicalFor
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    (parent child : Phi81Relation.CEStatement shape Commitment) : Prop where
  relationStructure : child.constraintSystem = parent.constraintSystem
  point : child.point = parent.point
  stage : child.stage = .fresh

/-- Forgetting and rematerializing a canonical child changes no paper field. -/
theorem materialize_ofStatement
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    (parent child : Phi81Relation.CEStatement shape Commitment)
    (canonical : CanonicalFor parent child) :
    PiDecChildPayload.materialize parent
        (PiDecChildPayload.ofStatement child) = child := by
  rcases parent with
    ⟨parentSystem, parentCommitment, parentPublicInput, parentPoint,
      parentEvaluations, parentStage⟩
  rcases child with
    ⟨childSystem, childCommitment, childPublicInput, childPoint,
      childEvaluations, childStage⟩
  rcases canonical with ⟨systemEq, pointEq, stageEq⟩
  cases systemEq
  cases pointEq
  cases stageEq
  rfl

/-- On the canonical child carrier, equality of the three retained payload
fields implies equality of the complete CE statements. -/
theorem eq_of_payload_eq
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    (parent left right : Phi81Relation.CEStatement shape Commitment)
    (leftCanonical : CanonicalFor parent left)
    (rightCanonical : CanonicalFor parent right)
    (payloadEq :
      PiDecChildPayload.ofStatement left =
        PiDecChildPayload.ofStatement right) :
    left = right := by
  calc
    left = PiDecChildPayload.materialize parent
        (PiDecChildPayload.ofStatement left) :=
      (materialize_ofStatement parent left leftCanonical).symm
    _ = PiDecChildPayload.materialize parent
        (PiDecChildPayload.ofStatement right) := by rw [payloadEq]
    _ = right := materialize_ofStatement parent right rightCanonical

/-- Canonicality of every child in one fixed-size family. -/
def CanonicalFamily
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment) : Prop :=
  ∀ child, CanonicalFor parent (children child)

/-- The actual `PiDEC.childrenOf` dataflow establishes canonicality by
construction. No acceptance equation or prover-carried certificate is needed
for these inherited fields. -/
theorem canonicalFamily_childrenOf
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params)
    (parent : Phi81Relation.CEStatement shape Commitment)
    (assignment : Assignment) :
    CanonicalFamily parent (PiDEC.childrenOf algebra parent assignment) := by
  intro child
  exact ⟨rfl, rfl, rfl⟩

/-- The existing canonical-opening authority predicate therefore supplies the
exact premise required by the reduced payload theorem. -/
theorem canonicalFamily_of_forOpening
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    {parent : Phi81Relation.CEStatement shape Commitment}
    {assignment : Assignment}
    {children : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
    (bound : PiDEC.CanonicalChildren.ForOpening algebra parent assignment children) :
    CanonicalFamily parent children := by
  rw [bound.childrenEq]
  exact canonicalFamily_childrenOf algebra parent assignment

/-- Strict public PiDEC acceptance also establishes the three inherited child
fields. This is useful for the next invocation, whose carried parent is
checked rather than trusted. -/
theorem canonicalFamily_of_accepted
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    {parent : Phi81Relation.CEStatement shape Commitment}
    {children : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
    (accepted : PiDEC.Accepted algebra {
      parent := parent
      children := children
    }) : CanonicalFamily parent children := by
  intro child
  exact ⟨accepted.sameStructure child, accepted.samePoint child,
    accepted.childFresh child⟩

/-- Exact ordered list of the three-field child payloads. -/
def payloadList
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment) :
    List (PiDecChildPayload shape Commitment) :=
  List.ofFn fun child => PiDecChildPayload.ofStatement (children child)

/-- Minimal exact paper carrier for one child family across different parent
contexts. The inherited structure and point occur once; the fresh stage is a
verifier-fixed constant and is not carried. -/
structure FamilyPayload
    (shape : Phi81Relation.Shape)
    (Commitment : Type uCommitment) where
  constraintSystem : Phi81Relation.Structure shape
  point : Phi81Relation.Point shape
  children : List (PiDecChildPayload shape Commitment)

/-- Project one strict PiDEC parent-plus-children view to its compact exact
paper authority carrier. -/
def familyPayload
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment) :
    FamilyPayload shape Commitment where
  constraintSystem := parent.constraintSystem
  point := parent.point
  children := payloadList children

private theorem payloadFunction_eq_of_list_eq
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    {left right : Fin count -> Phi81Relation.CEStatement shape Commitment}
    (same : payloadList left = payloadList right) :
    (fun child => PiDecChildPayload.ofStatement (left child)) =
      (fun child => PiDecChildPayload.ofStatement (right child)) := by
  funext child
  have entryEq := congrArg
    (fun values : List (PiDecChildPayload shape Commitment) =>
      values[child.val]?) same
  simpa [payloadList, child.isLt] using entryEq

/-- Equality of the ordered minimal payload list recovers the complete
canonical CE child family. -/
theorem family_eq_of_payloadList_eq
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (parent : Phi81Relation.CEStatement shape Commitment)
    (left right : Fin count -> Phi81Relation.CEStatement shape Commitment)
    (leftCanonical : CanonicalFamily parent left)
    (rightCanonical : CanonicalFamily parent right)
    (same : payloadList left = payloadList right) :
    left = right := by
  have payloadFunctions := payloadFunction_eq_of_list_eq same
  funext child
  apply eq_of_payload_eq parent (left child) (right child)
      (leftCanonical child) (rightCanonical child)
  exact congrFun payloadFunctions child

/-- Equality of the compact family carrier recovers the complete child vector
even when the two families were materialized from distinct parent values. -/
theorem children_eq_of_familyPayload_eq
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (leftParent rightParent : Phi81Relation.CEStatement shape Commitment)
    (left right : Fin count -> Phi81Relation.CEStatement shape Commitment)
    (leftCanonical : CanonicalFamily leftParent left)
    (rightCanonical : CanonicalFamily rightParent right)
    (same : familyPayload leftParent left = familyPayload rightParent right) :
    left = right := by
  have structureEq :
      leftParent.constraintSystem = rightParent.constraintSystem :=
    congrArg (fun payload => payload.constraintSystem) same
  have pointEq : leftParent.point = rightParent.point :=
    congrArg (fun payload => payload.point) same
  have payloadsEq : payloadList left = payloadList right :=
    congrArg (fun payload => payload.children) same
  have rightCanonicalForLeft : CanonicalFamily leftParent right := by
    intro child
    exact {
      relationStructure :=
        (rightCanonical child).relationStructure.trans structureEq.symm
      point := (rightCanonical child).point.trans pointEq.symm
      stage := (rightCanonical child).stage
    }
  exact family_eq_of_payloadList_eq leftParent left right leftCanonical
    rightCanonicalForLeft payloadsEq

/-- For two strictly accepted PiDEC views, equality of the compact family
carrier recovers both the exact ordered children and the recomposition-cache
parent. This is the semantic target for the recursive accumulator handle. -/
theorem parent_children_eq_of_familyPayload_eq
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
    (kPositive : 0 < params.k)
    (leftAccepted : PiDEC.Accepted algebra {
      parent := leftParent
      children := left
    })
    (rightAccepted : PiDEC.Accepted algebra {
      parent := rightParent
      children := right
    })
    (same : familyPayload leftParent left = familyPayload rightParent right) :
    leftParent = rightParent /\ left = right := by
  have childrenEq := children_eq_of_familyPayload_eq leftParent rightParent
    left right (canonicalFamily_of_accepted leftAccepted)
    (canonicalFamily_of_accepted rightAccepted) same
  exact ⟨PiDEC.Accepted.parent_eq_of_children_eq kPositive leftAccepted
      rightAccepted childrenEq,
    childrenEq⟩

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority
