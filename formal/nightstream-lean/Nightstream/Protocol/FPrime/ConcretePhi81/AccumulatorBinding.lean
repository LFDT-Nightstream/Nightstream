import Nightstream.Protocol.FPrime.AccumulatorBinding
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

/-!
Compact accumulator binding for one concrete Phi81 PiDEC family.

Assurance tier: model-level security partition.

Owns: compression of two proved compact child-family semantic carriers:
the exact public child payload; the smaller ordered-commitment carrier when
explicit valid child openings are available; and the point-plus-parent carrier
when both families have canonical child-opening authority. Equal handles reduce
to exact strict-PiDEC parent and child values or named serialization, hash, or
opening-binding failures.

Does not own: a concrete field serializer, Poseidon2 parameters or collision
bounds, Rust sidecar classification, Rust/R1CS refinement, costs, or row
removal.

Emits constraints: no.

Authority boundary: the full public-payload carrier is exact without private
opening assumptions. The commitment-only carrier is smaller, but is sufficient
only when both child families have explicit valid CE openings and unequal
openings are reduced to `Opening.BindingCollision`. Fresh stage is
verifier-fixed. Neither digest is authority unless recomputed from its carrier
and reduced through the corresponding failure partition. The parent-only
carrier additionally requires canonical child openings; public PiDEC
recomposition alone is insufficient.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.accumulator.family.payload` | exact common context plus ordered child-specific payloads | semantic input | `ChildPayloadAuthority.familyPayload` |
| `fprime.accumulator.family.digest` | compress the exact family carrier | security boundary | `familyDigest` |
| `fprime.accumulator.family.binding` | equal handles recover exact parent and children or name a failure | security reduction | `parent_children_eq_or_failure` |
| `fprime.accumulator.commitments.payload` | per-step point plus exact fixed-arity ordered child commitments | semantic input plus openings | `ChildCommitmentAuthority.commitmentFamilyPayload` |
| `fprime.accumulator.commitments.digest` | compress the smaller commitment carrier | security boundary | `commitmentFamilyDigest` |
| `fprime.accumulator.commitments.binding` | equal handles recover the strict PiDEC view or expose compression/opening failure | security reduction | `parent_children_eq_or_commitmentFailure` |
| `fprime.accumulator.canonical_parent.payload` | per-step point plus one canonically opened parent commitment | semantic input plus canonical openings | `CanonicalParentAuthority.canonicalParentPayload` |
| `fprime.accumulator.canonical_parent.digest` | compress the smallest conditional direct carrier | security boundary | `canonicalParentDigest` |
| `fprime.accumulator.canonical_parent.binding` | equal handles recover canonical parent and children or expose compression/opening failure | security reduction | `parent_children_eq_or_canonicalParentFailure` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

universe uAssignment uCommitment uEncoding uDigest

/-- Compact handle for the exact one-context-plus-children family carrier.
The concrete serializer remains a separate refinement obligation. -/
def familyDigest
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (FamilyPayload shape Commitment) Encoding Digest)
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment) :
    Digest :=
  Nightstream.Protocol.FPrime.AccumulatorBinding.claimDigest scheme
    (familyPayload parent children)

/-- Compact handle for the smaller point-plus-ordered-commitments carrier. Its
use requires separately bound relation structure and the explicit opening premises in
`parent_children_eq_or_commitmentFailure`. -/
def commitmentFamilyDigest
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CommitmentFamilyPayload shape Commitment count) Encoding Digest)
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment) :
    Digest :=
  Nightstream.Protocol.FPrime.AccumulatorBinding.claimDigest scheme
    (commitmentFamilyPayload parent children)

/-- Compact handle for one canonically opened parent family: one per-step
point and one combined parent commitment. -/
def canonicalParentDigest
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CanonicalParentPayload shape Commitment) Encoding Digest)
    (parent : Phi81Relation.CEStatement shape Commitment) : Digest :=
  Nightstream.Protocol.FPrime.AccumulatorBinding.claimDigest scheme
    (canonicalParentPayload parent)

/-- Exhaustive failure partition for the commitment-only family handle. -/
inductive CommitmentFamilyFailure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment)
    (params : GlobalParams)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CommitmentFamilyPayload shape Commitment params.k) Encoding Digest)
    (left : Fin params.k -> Phi81Relation.CEStatement shape Commitment) : Prop where
  | compression
      (failure :
        Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme)
  | childOpening
      (child : Fin params.k)
      (collision : Nonempty
        (Opening.BindingCollision semantics params.b
          (left child).commitment))

/-- Exhaustive failure partition for the canonical-parent handle. -/
inductive CanonicalParentFailure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment)
    (params : GlobalParams)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CanonicalParentPayload shape Commitment) Encoding Digest)
    (leftParent : Phi81Relation.CEStatement shape Commitment) : Prop where
  | compression
      (failure :
        Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme)
  | parentOpening
      (collision : Nonempty
        (Opening.BindingCollision semantics params.bigB
          leftParent.commitment))

/-- Binding reduction for the complete strict PiDEC view. Equal compact
handles recover both parent and ordered children, or expose the exact generic
serialization/hash failure. -/
theorem parent_children_eq_or_failure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (FamilyPayload shape Commitment) Encoding Digest)
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
    (sameDigest :
      familyDigest scheme leftParent left =
        familyDigest scheme rightParent right) :
    (leftParent = rightParent /\ left = right) \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases Nightstream.Protocol.FPrime.AccumulatorBinding.claim_eq_or_failure
      scheme (familyPayload leftParent left)
      (familyPayload rightParent right) sameDigest with payloadEq | failure
  · exact Or.inl (parent_children_eq_of_familyPayload_eq kPositive
      leftAccepted rightAccepted payloadEq)
  · exact Or.inr failure

/-- Binding reduction for the smaller ordered-commitment handle. The current
and previous child openings stay explicit: equal handles recover the complete
strict PiDEC views only outside the generic serializer/hash failure and one
fresh-bound child-opening collision. -/
theorem parent_children_eq_or_commitmentFailure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CommitmentFamilyPayload shape Commitment params.k) Encoding Digest)
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
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
    (sameDigest :
      commitmentFamilyDigest scheme leftParent left =
        commitmentFamilyDigest scheme rightParent right) :
    (leftParent = rightParent ∧ left = right) ∨
      CommitmentFamilyFailure semantics params scheme left := by
  rcases Nightstream.Protocol.FPrime.AccumulatorBinding.claim_eq_or_failure
      scheme (commitmentFamilyPayload leftParent left)
      (commitmentFamilyPayload rightParent right) sameDigest with
    payloadEq | failure
  · rcases parent_children_eq_or_freshBindingCollision kPositive
        leftAssignments rightAssignments leftAccepted rightAccepted
        leftValid rightValid sameStructure payloadEq with
      exactView | ⟨child, collision⟩
    · exact Or.inl exactView
    · exact Or.inr (.childOpening child collision)
  · exact Or.inr (.compression failure)

/-- Binding reduction for the smallest conditional direct carrier. Both views
must already have canonical child-opening authority; public PiDEC acceptance
cannot replace this premise. -/
theorem parent_children_eq_or_canonicalParentFailure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Assignment : Type uAssignment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {semantics : RelationSemantics
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra
      (Phi81Relation.Structure shape) Assignment
      (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
      Phi81Relation.Evaluation Commitment semantics params}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (CanonicalParentPayload shape Commitment) Encoding Digest)
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
    {leftAssignment rightAssignment : Assignment}
    (leftCanonical : PiDEC.CanonicalChildren.ForOpening algebra leftParent
      leftAssignment left)
    (rightCanonical : PiDEC.CanonicalChildren.ForOpening algebra rightParent
      rightAssignment right)
    (sameStructure :
      leftParent.constraintSystem = rightParent.constraintSystem)
    (sameDigest :
      canonicalParentDigest scheme leftParent =
        canonicalParentDigest scheme rightParent) :
    (leftParent = rightParent ∧ left = right) ∨
      CanonicalParentFailure semantics params scheme leftParent := by
  rcases Nightstream.Protocol.FPrime.AccumulatorBinding.claim_eq_or_failure
      scheme (canonicalParentPayload leftParent)
      (canonicalParentPayload rightParent) sameDigest with
    payloadEq | failure
  · rcases Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority.parent_children_eq_or_bindingCollision
        leftCanonical rightCanonical sameStructure payloadEq with
      exactView | collision
    · exact Or.inl exactView
    · exact Or.inr (.parentOpening collision)
  · exact Or.inr (.compression failure)

end Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding
