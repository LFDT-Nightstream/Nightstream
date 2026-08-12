import Nightstream.Protocol.NebulaV2.IdealSequence

set_option autoImplicit false

namespace tests.NebulaV2IdealSequence

open Nightstream.Protocol.NebulaV2.IdealSequence
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.SequenceBinding

abbrev Sequence := FramedSequence Nat Nat Nat

def chainRoot : Sequence → Sequence := id

def sequence (domain : LaneDomain) : Sequence :=
  { profile := 7
    plan := 9
    domain := domain
    commitments := fun index => index.val }

def lane (role : Role) : LaneCheck chainRoot 7 9 role where
  precommit :=
    { sequence := sequence role.domain
      committedRoot := sequence role.domain
      rootCorrect := rfl }
  replay :=
    { sequence := sequence role.domain
      seenRoot := sequence role.domain
      rootCorrect := rfl }
  closeRoot := rfl
  precommitProfile := rfl
  precommitPlan := rfl
  precommitDomain := rfl
  replayProfile := rfl
  replayPlan := rfl
  replayDomain := rfl

def checks : Checks chainRoot 7 9 where
  operations := lane .operations
  initialSnapshot := lane .initialSnapshot
  finalSnapshot := lane .finalSnapshot

theorem identity_chain_has_no_collision : ¬ RootCollision chainRoot := by
  rintro ⟨left, right, different, equalRoot⟩
  exact different equalRoot

theorem all_three_sequences_are_exact : checks.Exact := by
  rcases checks.exact_or_collision with exact | collision
  · exact exact
  · exact False.elim (identity_chain_has_no_collision collision)

theorem all_three_seen_roots_equal_their_precommits :
    checks.seenRoots = checks.committedRoots :=
  checks.roots_equal

end tests.NebulaV2IdealSequence
