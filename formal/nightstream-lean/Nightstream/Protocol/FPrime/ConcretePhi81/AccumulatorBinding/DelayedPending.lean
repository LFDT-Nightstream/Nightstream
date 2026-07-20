import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DelayedBlockLane

/-!
Exact-child accumulator binding for one delayed packed-`yZcol` value.

Assurance tier: model-level security partition.

Owns: the compact payload consisting of the existing exact ordered-child
family plus one optional production delayed value; recomputation of its
abstract accumulator handle; and reduction of equal handles to exact child
and pending equality or the generic serialization/hash failure.

Does not own: pending-state acceptance, one-fold continuity, a concrete
serializer, Poseidon2, commitment opening security, Rust/R1CS refinement,
rows, costs, or row removal.

Emits constraints: no.

Authority boundary: `pendingFamilyDigest` is compression, never authority.
It binds the child family and pending value only when both sides recompute it
from their complete typed payloads and the conclusion excludes the explicitly
returned `BindingFailure`. The checked `Pi_RLC` parent remains outside this
payload and must still be justified by strict `Pi_DEC` recomposition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.accumulator.delayed.children` | retain the exact shared context and ordered child payloads | direct dataflow | `PendingFamilyPayload.family` |
| `fprime.accumulator.delayed.pending` | retain absence or exactly one old block point and 54-lane aggregate | direct dataflow | `PendingFamilyPayload.pending` |
| `fprime.accumulator.delayed.digest` | recompute the compact handle from the complete typed payload | security boundary | `pendingFamilyDigest`, `StateBinds` |
| `fprime.accumulator.delayed.binding` | equal recomputed handles recover exact children and pending state or name a failure | derived/security boundary | `children_pending_eq_or_failure`, `children_pending_eq_or_failure_of_stateBinding` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildPayloadAuthority

universe uAssignment uCommitment uEncoding uDigest

/-- Exact public accumulator payload for the delayed route. The existing
family carrier binds common structure and point plus every ordered
child-specific payload. The optional field is `none` only at the base or for
an accumulator that has not yet produced a delayed value. -/
structure PendingFamilyPayload
    (shape : Phi81Relation.Shape)
    (Commitment : Type uCommitment) where
  family : FamilyPayload shape Commitment
  pending : Option ProductionDelayedBlockLane

/-- Project one strict parent/child carrier and its delayed state into the
sole payload compressed by the pending-aware accumulator handle. -/
def pendingFamilyPayload
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {count : Nat}
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment)
    (pending : Option ProductionDelayedBlockLane) :
    PendingFamilyPayload shape Commitment where
  family := familyPayload parent children
  pending := pending

/-- Compact handle for the complete exact-child-plus-pending payload. This is
an abstract compression boundary; no concrete hash gains authority here. -/
def pendingFamilyDigest
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload shape Commitment) Encoding Digest)
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment)
    (pending : Option ProductionDelayedBlockLane) : Digest :=
  Nightstream.Protocol.FPrime.AccumulatorBinding.claimDigest scheme
    (pendingFamilyPayload parent children pending)

/-- A state coordinate binds the delayed accumulator only through exact
recomputation from the complete typed payload. -/
def StateBinds
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload shape Commitment) Encoding Digest)
    (stateDigest : Digest)
    (parent : Phi81Relation.CEStatement shape Commitment)
    (children : Fin count -> Phi81Relation.CEStatement shape Commitment)
    (pending : Option ProductionDelayedBlockLane) : Prop :=
  stateDigest = pendingFamilyDigest scheme parent children pending

/-- Equal recomputed handles bind both the exact ordered children and the
optional pending value, modulo precisely the generic encoding/hash failure.

The canonical-family premises justify omission of the child fields inherited
from each parent. No parent equality follows here: the parent is a checked
recomposition cache and remains separately owned by `Pi_DEC`. -/
theorem children_pending_eq_or_failure
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload shape Commitment) Encoding Digest)
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin count -> Phi81Relation.CEStatement shape Commitment}
    {leftPending rightPending : Option ProductionDelayedBlockLane}
    (leftCanonical : CanonicalFamily leftParent left)
    (rightCanonical : CanonicalFamily rightParent right)
    (sameDigest :
      pendingFamilyDigest scheme leftParent left leftPending =
        pendingFamilyDigest scheme rightParent right rightPending) :
    (left = right /\ leftPending = rightPending) \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases Nightstream.Protocol.FPrime.AccumulatorBinding.claim_eq_or_failure
      scheme (pendingFamilyPayload leftParent left leftPending)
      (pendingFamilyPayload rightParent right rightPending) sameDigest with
    payloadEq | failure
  · have familyEq :
        familyPayload leftParent left = familyPayload rightParent right :=
      congrArg PendingFamilyPayload.family payloadEq
    have pendingEq : leftPending = rightPending :=
      congrArg PendingFamilyPayload.pending payloadEq
    exact Or.inl ⟨children_eq_of_familyPayload_eq leftParent rightParent
      left right leftCanonical rightCanonical familyEq, pendingEq⟩
  · exact Or.inr failure

/-- The same binding reduction phrased at the actual carried state
coordinate. Both sides must recompute that coordinate from their complete
payload; equality of two unqualified digests is deliberately insufficient. -/
theorem children_pending_eq_or_failure_of_stateBinding
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {count : Nat}
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload shape Commitment) Encoding Digest)
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin count -> Phi81Relation.CEStatement shape Commitment}
    {leftPending rightPending : Option ProductionDelayedBlockLane}
    {leftStateDigest rightStateDigest : Digest}
    (leftCanonical : CanonicalFamily leftParent left)
    (rightCanonical : CanonicalFamily rightParent right)
    (leftBinds :
      StateBinds scheme leftStateDigest leftParent left leftPending)
    (rightBinds :
      StateBinds scheme rightStateDigest rightParent right rightPending)
    (sameStateDigest : leftStateDigest = rightStateDigest) :
    (left = right /\ leftPending = rightPending) \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  apply children_pending_eq_or_failure scheme leftCanonical rightCanonical
  exact leftBinds.symm.trans (sameStateDigest.trans rightBinds)

/-- Strict `Pi_DEC` acceptance additionally recovers the checked parent
cache. This is a convenience composition theorem; child/pending binding does
not rely on a parent-only handle. -/
theorem parent_children_pending_eq_or_failure
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
      (PendingFamilyPayload shape Commitment) Encoding Digest)
    {leftParent rightParent : Phi81Relation.CEStatement shape Commitment}
    {left right : Fin params.k -> Phi81Relation.CEStatement shape Commitment}
    {leftPending rightPending : Option ProductionDelayedBlockLane}
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
      pendingFamilyDigest scheme leftParent left leftPending =
        pendingFamilyDigest scheme rightParent right rightPending) :
    (leftParent = rightParent /\ left = right /\
        leftPending = rightPending) \/
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases children_pending_eq_or_failure scheme
      (canonicalFamily_of_accepted leftAccepted)
      (canonicalFamily_of_accepted rightAccepted) sameDigest with
    exactPayload | failure
  · have parentEq := PiDEC.Accepted.parent_eq_of_children_eq kPositive
      leftAccepted rightAccepted exactPayload.1
    exact Or.inl ⟨parentEq, exactPayload.1, exactPayload.2⟩
  · exact Or.inr failure

end Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
