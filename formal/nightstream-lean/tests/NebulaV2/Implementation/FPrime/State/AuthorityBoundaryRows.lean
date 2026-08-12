import Nightstream.Implementation.NebulaV2.FPrime.State.AuthorityBoundaryRows

/-! Focused regressions for the exact cross-invocation V2 state boundary. -/

namespace NightstreamTests.NebulaV2StateAuthorityBoundaryRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows

example (layout : Layout) : (rows layout).length = 4 :=
  rows_length_exact layout

example {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    Same outgoing incoming ∨ Failure :=
  boundary.sound

example {outgoing incoming : Authority}
    (boundary : Boundary outgoing incoming) :
    outgoing.digest = incoming.digest :=
  boundary.digest_eq

example {first : Invocation} {rest : List Invocation}
    (chain : CandidateChain first rest) :
    ExactChain first rest ∨ Failure :=
  candidate_sound_or_collision chain

namespace ConstantDigestCountermodel

/-- Digest equality alone cannot be an authority theorem. A constant digest
accepts two different states. The production theorem needs both the exact
four-lane boundary rows and the named collision branch. -/
inductive ToyState where
  | left
  | right
deriving DecidableEq

def digest (_state : ToyState) : Nat := 0

theorem distinct_states_have_equal_digest :
    ToyState.left ≠ ToyState.right ∧
      digest ToyState.left = digest ToyState.right := by
  decide

end ConstantDigestCountermodel

end NightstreamTests.NebulaV2StateAuthorityBoundaryRows
