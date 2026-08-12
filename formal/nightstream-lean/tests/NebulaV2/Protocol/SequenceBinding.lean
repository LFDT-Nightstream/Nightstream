import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2SequenceBinding

open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.SequenceBinding

abbrev Sequence := FramedSequence Bool Unit Bool

def allFalse : Fin claimsPerSegment → Bool := fun _ => false

def committedSequence : Sequence where
  profile := false
  plan := ()
  domain := .operations
  commitments := allFalse

def honestRoot (sequence : Sequence) : Sequence := sequence

def precommit : KnownPrecommit honestRoot where
  sequence := committedSequence
  committedRoot := committedSequence
  rootCorrect := rfl

def replay : CheckedReplay honestRoot where
  sequence := committedSequence
  seenRoot := committedSequence
  rootCorrect := rfl

theorem identity_root_has_no_collision : ¬ RootCollision honestRoot := by
  rintro ⟨left, right, different, equalRoot⟩
  exact different equalRoot

theorem honest_close_binds_every_position :
    replay.sequence.commitments = precommit.sequence.commitments :=
  close_binds_every_commitment identity_root_has_no_collision
    precommit replay rfl

/- If the chain root omits the profile frame, equal roots do not bind the
profile. This is a deterministic collision, even when every commitment is the
same. -/
namespace MissingFrame

def left : Sequence where
  profile := false
  plan := ()
  domain := .memory
  commitments := allFalse

def right : Sequence where
  profile := true
  plan := ()
  domain := .memory
  commitments := allFalse

def unframedRoot (sequence : Sequence) : Fin claimsPerSegment → Bool :=
  sequence.commitments

theorem sequences_differ : left ≠ right := by
  intro equal
  have profileEqual := congrArg FramedSequence.profile equal
  change false = true at profileEqual
  contradiction

theorem omitted_profile_creates_collision : RootCollision unframedRoot := by
  exact ⟨left, right, sequences_differ, rfl⟩

end MissingFrame

end tests.NebulaV2SequenceBinding
