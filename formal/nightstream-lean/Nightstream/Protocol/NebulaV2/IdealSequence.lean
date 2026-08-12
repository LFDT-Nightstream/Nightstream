import Nightstream.Protocol.NebulaV2.FPrime
import Nightstream.Protocol.NebulaV2.CommitmentBundle
import Nightstream.Protocol.NebulaV2.SequenceBinding

/-!
Contract: ideal-verifier composition of all three V2 prechallenge lanes.

Assurance tier: model-level and cryptographic-reduction boundary.

Owns the three fixed lane roles, verifier-owned profile and plan checks, exact
precommit-to-replay root checks, and the reduction to three exact framed
sequences or one explicit root collision.

Does not own Poseidon2 collision or late-preimage security, commitment
openings, circuit rows, or transcript probability.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.IdealSequence

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.Protocol.NebulaV2.SequenceBinding

inductive Role where
  | operations
  | initialSnapshot
  | finalSnapshot
deriving DecidableEq, Repr

def Role.domain : Role → LaneDomain
  | .operations => .operations
  | .initialSnapshot => .memory
  | .finalSnapshot => .memory

def Role.component : Role → Component
  | .operations => .operations
  | .initialSnapshot => .initialSnapshot
  | .finalSnapshot => .finalSnapshot

/-- One checked lane. Both decoded sequences must carry the verifier-owned
profile, plan, and role domain. -/
structure LaneCheck
    {Profile Plan Commitment Digest : Type}
    (chainRoot : FramedSequence Profile Plan Commitment → Digest)
    (profile : Profile) (plan : Plan) (role : Role) where
  precommit : KnownPrecommit chainRoot
  replay : CheckedReplay chainRoot
  closeRoot : replay.seenRoot = precommit.committedRoot
  precommitProfile : precommit.sequence.profile = profile
  precommitPlan : precommit.sequence.plan = plan
  precommitDomain : precommit.sequence.domain = role.domain
  replayProfile : replay.sequence.profile = profile
  replayPlan : replay.sequence.plan = plan
  replayDomain : replay.sequence.domain = role.domain

namespace LaneCheck

theorem exact_or_collision
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan} {role : Role}
    (check : LaneCheck chainRoot profile plan role) :
    check.replay.sequence = check.precommit.sequence ∨
      RootCollision chainRoot :=
  close_binds_exact_sequence_or_collision
    check.precommit check.replay check.closeRoot

theorem exact_of_noCollision
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan} {role : Role}
    (check : LaneCheck chainRoot profile plan role)
    (noCollision : ¬ RootCollision chainRoot) :
    check.replay.sequence = check.precommit.sequence := by
  rcases check.exact_or_collision with exact | collision
  · exact exact
  · exact False.elim (noCollision collision)

end LaneCheck

/-- All three V2 roots are checked as one protocol object. Initial and final
snapshots share the memory lane function but retain distinct roles here. -/
structure Checks
    {Profile Plan Commitment Digest : Type}
    (chainRoot : FramedSequence Profile Plan Commitment → Digest)
    (profile : Profile) (plan : Plan) where
  operations : LaneCheck chainRoot profile plan .operations
  initialSnapshot : LaneCheck chainRoot profile plan .initialSnapshot
  finalSnapshot : LaneCheck chainRoot profile plan .finalSnapshot

namespace Checks

def lane
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) :
    (role : Role) → LaneCheck chainRoot profile plan role
  | .operations => checks.operations
  | .initialSnapshot => checks.initialSnapshot
  | .finalSnapshot => checks.finalSnapshot

def committedRoots
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) : Roots Digest :=
  { operations := checks.operations.precommit.committedRoot
    initialSnapshot := checks.initialSnapshot.precommit.committedRoot
    finalSnapshot := checks.finalSnapshot.precommit.committedRoot }

def seenRoots
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) : Roots Digest :=
  { operations := checks.operations.replay.seenRoot
    initialSnapshot := checks.initialSnapshot.replay.seenRoot
    finalSnapshot := checks.finalSnapshot.replay.seenRoot }

def Exact
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) : Prop :=
  checks.operations.replay.sequence =
      checks.operations.precommit.sequence ∧
    checks.initialSnapshot.replay.sequence =
      checks.initialSnapshot.precommit.sequence ∧
    checks.finalSnapshot.replay.sequence =
      checks.finalSnapshot.precommit.sequence

theorem roots_equal
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) :
    checks.seenRoots = checks.committedRoots := by
  apply Roots.ext <;> simp [seenRoots, committedRoots,
    checks.operations.closeRoot, checks.initialSnapshot.closeRoot,
    checks.finalSnapshot.closeRoot]

theorem exact_or_collision
    {Profile Plan Commitment Digest : Type}
    {chainRoot : FramedSequence Profile Plan Commitment → Digest}
    {profile : Profile} {plan : Plan}
    (checks : Checks chainRoot profile plan) :
    checks.Exact ∨ RootCollision chainRoot := by
  by_cases collision : RootCollision chainRoot
  · exact Or.inr collision
  · exact Or.inl
      ⟨checks.operations.exact_of_noCollision collision,
        checks.initialSnapshot.exact_of_noCollision collision,
        checks.finalSnapshot.exact_of_noCollision collision⟩

end Checks

end Nightstream.Protocol.NebulaV2.IdealSequence
