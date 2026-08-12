import Nightstream.Protocol.NebulaV2.FullClaim

set_option autoImplicit false

namespace tests.NebulaV2FullClaim

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.FullClaim

def schema : Schema where
  CcsPublic := Unit
  ApplicationPublic := Bool
  CommitmentBundle := Unit
  RecursiveState := Unit
  NifsProof := Unit

def roots : Roots Unit := ⟨(), (), ()⟩

def suffix : ClaimSuffix Unit Unit Unit where
  segmentIndex := 0
  stepIndex := ⟨0, by decide⟩
  timestampIn := 0
  timestampOut := 0
  segmentStartTimestamp := 0
  segmentEndTimestamp := 0
  activeAccessCount := 0
  challenge := ()
  dPre := roots
  dSeenBefore := roots
  dSeenAfter := roots
  productsBefore := ()
  productsAfter := ()

def left : Claim schema Unit Unit Unit where
  profile := Profile.v2
  ccsPublic := ()
  applicationPublic := false
  commitmentBundle := ()
  recursiveState := ()
  memory := suffix

def right : Claim schema Unit Unit Unit where
  profile := Profile.v2
  ccsPublic := ()
  applicationPublic := true
  commitmentBundle := ()
  recursiveState := ()
  memory := suffix

/-- A suffix is not an injective representation of a full claim. The full-run
model is safe because its verifier and transition share one `Verified` value. -/
theorem distinct_full_claims_can_share_one_suffix :
    left ≠ right ∧ left.memory = right.memory := by
  constructor
  · intro equal
    have applicationEqual := congrArg Claim.applicationPublic equal
    change false = true at applicationEqual
    contradiction
  · rfl

end tests.NebulaV2FullClaim
