import Nightstream.Protocol.NebulaV2.Lifecycle

/-!
Contract: deterministic boundary for V2 prechallenge chain binding.

Assurance tier: model-level and cryptographic-reduction boundary.

Owns the fixed-length, profile-bound, plan-bound, domain-bound lane sequence,
known precommitment witness, checked replay witness, and the reduction from an
equal close root to either exact sequence equality or a named root collision.

Does not prove Poseidon2 collision resistance, known-preimage extraction in the
random-oracle model, compact-token binding, canonical decoding, or circuit
refinement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.SequenceBinding

open Nightstream.Protocol.NebulaV2.Lifecycle

/-- Initial and final snapshots intentionally share the `memory` chain domain.
Their use as initial or final is bound later by the challenge transcript. -/
inductive LaneDomain where
  | operations
  | memory
deriving DecidableEq, Repr

/-- The commitment count is structural. A missing or repeated position cannot
be represented as a shorter fixed-length sequence. -/
structure FramedSequence
    (Profile Plan Commitment : Type) where
  profile : Profile
  plan : Plan
  domain : LaneDomain
  commitments : Fin claimsPerSegment → Commitment

@[ext]
theorem FramedSequence.ext
    {Profile Plan Commitment : Type}
    {left right : FramedSequence Profile Plan Commitment}
    (profileEqual : left.profile = right.profile)
    (planEqual : left.plan = right.plan)
    (domainEqual : left.domain = right.domain)
    (commitmentsEqual : left.commitments = right.commitments) :
    left = right := by
  cases left with
  | mk leftProfile leftPlan leftDomain leftCommitments =>
      cases right with
      | mk rightProfile rightPlan rightDomain rightCommitments =>
          change leftProfile = rightProfile at profileEqual
          change leftPlan = rightPlan at planEqual
          change leftDomain = rightDomain at domainEqual
          change leftCommitments = rightCommitments at commitmentsEqual
          cases profileEqual
          cases planEqual
          cases domainEqual
          cases commitmentsEqual
          rfl

/-- A collision includes framing differences, not only leaf differences. -/
def RootCollision
    {Profile Plan Commitment Digest : Type}
    (chainRoot : FramedSequence Profile Plan Commitment → Digest) : Prop :=
  ∃ left right,
    left ≠ right ∧ chainRoot left = chainRoot right

/-- This is the witness that the ROM reduction must extract before it releases
the memory challenge. A prover-selected digest alone is not enough. -/
structure KnownPrecommit
    {Profile Plan Commitment Digest : Type}
    (chainRoot : FramedSequence Profile Plan Commitment → Digest) where
  sequence : FramedSequence Profile Plan Commitment
  committedRoot : Digest
  rootCorrect : chainRoot sequence = committedRoot

/-- The checked phase recomputes its root from the exact verified bundle
sequence. -/
structure CheckedReplay
    {Profile Plan Commitment Digest : Type}
    (chainRoot : FramedSequence Profile Plan Commitment → Digest) where
  sequence : FramedSequence Profile Plan Commitment
  seenRoot : Digest
  rootCorrect : chainRoot sequence = seenRoot

theorem equal_root_implies_equal_sequence_or_collision
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {left right : FramedSequence Profile Plan Commitment}
    (equalRoot : chainRoot left = chainRoot right) :
    left = right ∨ RootCollision chainRoot := by
  by_cases equalSequence : left = right
  · exact Or.inl equalSequence
  · exact Or.inr ⟨left, right, equalSequence, equalRoot⟩

/-- If a sequence was extracted before the challenge and close equates the
replayed root with that committed root, the two complete framed sequences are
equal unless the root function has a collision. -/
theorem close_binds_exact_sequence_or_collision
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    (precommit : KnownPrecommit chainRoot)
    (replay : CheckedReplay chainRoot)
    (closeRoot : replay.seenRoot = precommit.committedRoot) :
    replay.sequence = precommit.sequence ∨ RootCollision chainRoot := by
  apply equal_root_implies_equal_sequence_or_collision
  calc
    chainRoot replay.sequence = replay.seenRoot := replay.rootCorrect
    _ = precommit.committedRoot := closeRoot
    _ = chainRoot precommit.sequence := precommit.rootCorrect.symm

theorem close_binds_every_commitment
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    (noCollision : ¬ RootCollision chainRoot)
    (precommit : KnownPrecommit chainRoot)
    (replay : CheckedReplay chainRoot)
    (closeRoot : replay.seenRoot = precommit.committedRoot) :
    replay.sequence.commitments = precommit.sequence.commitments := by
  rcases close_binds_exact_sequence_or_collision precommit replay closeRoot with
    equalSequence | collision
  · exact congrArg FramedSequence.commitments equalSequence
  · exact False.elim (noCollision collision)

theorem close_binds_profile_plan_and_domain
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    (noCollision : ¬ RootCollision chainRoot)
    (precommit : KnownPrecommit chainRoot)
    (replay : CheckedReplay chainRoot)
    (closeRoot : replay.seenRoot = precommit.committedRoot) :
    replay.sequence.profile = precommit.sequence.profile ∧
      replay.sequence.plan = precommit.sequence.plan ∧
      replay.sequence.domain = precommit.sequence.domain := by
  rcases close_binds_exact_sequence_or_collision precommit replay closeRoot with
    equalSequence | collision
  · exact
      ⟨congrArg FramedSequence.profile equalSequence,
        congrArg FramedSequence.plan equalSequence,
        congrArg FramedSequence.domain equalSequence⟩
  · exact False.elim (noCollision collision)

end Nightstream.Protocol.NebulaV2.SequenceBinding
