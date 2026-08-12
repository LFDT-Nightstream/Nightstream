import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2FPrime

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle

def balanced (products : Nat) : Prop := products = 0

def headers : Roots Nat := ⟨0, 0, 0⟩
def chainHeaders : ChainHeaders Nat := ⟨0, 0⟩
def precommit : Roots Nat := ⟨10, 20, 30⟩

def activeFirst : ActiveCarry Nat Nat Nat :=
  { segmentIndex := 0
    stepIndex := ⟨0, by decide⟩
    globalTimestamp := 0
    segmentStartTimestamp := 0
    segmentActiveAccessCount := 2
    segmentEndTimestamp := 2
    challenge := 77
    products := 0
    dPre := precommit
    dSeen := headers
    memoryRoot := 20 }

def firstClaim : ClaimSuffix Nat Nat Nat :=
  { segmentIndex := 0
    stepIndex := ⟨0, by decide⟩
    timestampIn := 0
    timestampOut := 1
    segmentStartTimestamp := 0
    segmentEndTimestamp := 2
    activeAccessCount := 1
    challenge := 77
    dPre := precommit
    dSeenBefore := headers
    dSeenAfter := ⟨1, 1, 1⟩
    productsBefore := 0
    productsAfter := 0 }

def firstMatches : MatchesActive activeFirst firstClaim where
  activeWellFormed := by
    exact ⟨by decide, by decide, rfl, by decide, by decide, by decide⟩
  segmentIndex := rfl
  stepIndex := rfl
  timestampIn := rfl
  segmentStartTimestamp := rfl
  segmentEndTimestamp := rfl
  challenge := rfl
  dPre := rfl
  dSeen := rfl
  products := rfl
  timestampAdvance := rfl
  activeCountBound := by decide
  timestampWithinDeclaredEnd := by decide
  timestampInRange := by decide
  timestampOutRange := by decide

def wrongChallengeClaim : ClaimSuffix Nat Nat Nat :=
  { firstClaim with challenge := 78 }

theorem changed_challenge_cannot_match_active_carry :
    ¬ MatchesActive activeFirst wrongChallengeClaim := by
  intro agreement
  have exactChallenge := agreement.binds_challenge_and_segment_bounds.1
  change 78 = 77 at exactChallenge
  omega

def wrongStartTimestampClaim : ClaimSuffix Nat Nat Nat :=
  { firstClaim with segmentStartTimestamp := 1 }

theorem changed_segment_start_cannot_match_active_carry :
    ¬ MatchesActive activeFirst wrongStartTimestampClaim := by
  intro agreement
  have exactStart := agreement.binds_challenge_and_segment_bounds.2.1
  change 1 = 0 at exactStart
  omega

def wrongEndTimestampClaim : ClaimSuffix Nat Nat Nat :=
  { firstClaim with segmentEndTimestamp := 3 }

theorem changed_segment_end_cannot_match_active_carry :
    ¬ MatchesActive activeFirst wrongEndTimestampClaim := by
  intro agreement
  have exactEnd := agreement.binds_challenge_and_segment_bounds.2.2
  change 3 = 2 at exactEnd
  omega

def interiorTransition :
    Consumes balanced (.active activeFirst) firstClaim
      (.active (interiorCarry activeFirst firstClaim (by decide))) :=
  .interior firstMatches (by decide)

def activeLast : ActiveCarry Nat Nat Nat :=
  { segmentIndex := 0
    stepIndex := ⟨1087, by decide⟩
    globalTimestamp := 5
    segmentStartTimestamp := 0
    segmentActiveAccessCount := 6
    segmentEndTimestamp := 6
    challenge := 77
    products := 0
    dPre := precommit
    dSeen := ⟨9, 9, 9⟩
    memoryRoot := 20 }

def lastClaim : ClaimSuffix Nat Nat Nat :=
  { segmentIndex := 0
    stepIndex := ⟨1087, by decide⟩
    timestampIn := 5
    timestampOut := 6
    segmentStartTimestamp := 0
    segmentEndTimestamp := 6
    activeAccessCount := 1
    challenge := 77
    dPre := precommit
    dSeenBefore := ⟨9, 9, 9⟩
    dSeenAfter := precommit
    productsBefore := 0
    productsAfter := 0 }

def lastMatches : MatchesActive activeLast lastClaim where
  activeWellFormed := by
    exact ⟨by decide, by decide, rfl, by decide, by decide, by decide⟩
  segmentIndex := rfl
  stepIndex := rfl
  timestampIn := rfl
  segmentStartTimestamp := rfl
  segmentEndTimestamp := rfl
  challenge := rfl
  dPre := rfl
  dSeen := rfl
  products := rfl
  timestampAdvance := rfl
  activeCountBound := by decide
  timestampWithinDeclaredEnd := by decide
  timestampInRange := by decide
  timestampOutRange := by decide

def lastChecks : CloseChecks balanced activeLast lastClaim where
  seenEqualsPrecommit := rfl
  initialEqualsMemory := rfl
  productsBalanced := rfl
  timestampEqualsDeclaredEnd := rfl

def closeTransition :
    Consumes balanced (.active activeLast) lastClaim
      (.closed (closedCarryAfter activeLast lastClaim)) :=
  .close lastMatches rfl lastChecks

theorem close_sets_exact_boundary_state :
    closedCarryAfter activeLast lastClaim =
      ({ segmentIndex := 1
         globalTimestamp := 6
         memoryRoot := 30 } : ClosedCarry Nat) :=
  rfl

theorem close_requires_last_balance_roots_and_timestamp :
    lastClaim.stepIndex.val + 1 = claimsPerSegment ∧
      CloseChecks balanced activeLast lastClaim ∧
      closedCarryAfter activeLast lastClaim =
        closedCarryAfter activeLast lastClaim :=
  active_to_closed_requires_all_close_checks closeTransition

theorem close_exposes_initial_memory_authority :
    lastClaim.dSeenAfter.initialSnapshot = activeLast.memoryRoot :=
  (active_to_closed_requires_all_close_checks closeTransition).2.1.initialEqualsMemory

theorem close_exposes_declared_end_timestamp :
    lastClaim.timestampOut = activeLast.segmentEndTimestamp :=
  (active_to_closed_requires_all_close_checks
    closeTransition).2.1.timestampEqualsDeclaredEnd

/- A detached boundary field can agree with carried memory while the replayed
snapshot chain disagrees. It can also select a next memory root that the chain
did not produce. The V2 close state therefore projects both roots directly
from `dSeenAfter`. -/
namespace DetachedBoundaryFields

theorem false_boundary_countermodel :
    ∃ (seen pre : Roots Nat) (detachedInitial detachedFinal memory : Nat),
      seen = pre ∧
      detachedInitial = memory ∧
      seen.initialSnapshot ≠ memory ∧
      detachedFinal ≠ seen.finalSnapshot := by
  exact ⟨⟨10, 20, 30⟩, ⟨10, 20, 30⟩, 99, 88, 99,
    rfl, rfl, by decide, by decide⟩

end DetachedBoundaryFields

def verifiesLast (claim : ClaimSuffix Nat Nat Nat) : Prop := claim = lastClaim

def verifiedClose :
    VerifiedTransition verifiesLast balanced (.active activeLast) lastClaim
      (.closed (closedCarryAfter activeLast lastClaim)) where
  verified := rfl
  consumes := closeTransition

def oneClaimVerifiedRun :
    VerifiedRun verifiesLast balanced (.active activeLast) [lastClaim]
      (.closed (closedCarryAfter activeLast lastClaim)) :=
  .cons verifiedClose (.nil _)

theorem last_step_run_consumes_exactly_one_claim :
    [lastClaim].length = claimsPerSegment - activeLast.stepIndex.val :=
  VerifiedRun.to_closed_has_exact_remaining_length oneClaimVerifiedRun

theorem closing_run_exposes_the_checked_balance :
    ∃ finalClaim ∈ [lastClaim], balanced finalClaim.productsAfter :=
  VerifiedRun.to_closed_has_balanced_products oneClaimVerifiedRun

def acceptsAnyProducts (_products : Nat) : Prop := True

def weakenedRun :
    VerifiedRun verifiesLast acceptsAnyProducts (.active activeLast)
      [lastClaim] (.closed (closedCarryAfter activeLast lastClaim)) :=
  oneClaimVerifiedRun.mono (by
    intro products balancedProducts
    exact True.intro)

theorem strengthened_run_still_closes :
    ∃ finalClaim ∈ [lastClaim], acceptsAnyProducts finalClaim.productsAfter :=
  weakenedRun.to_closed_has_balanced_products

theorem a_closed_run_cannot_have_post_close_claims
    {claims : List (ClaimSuffix Nat Nat Nat)}
    {after : Carry Nat Nat Nat}
    (run : VerifiedRun verifiesLast balanced
      (.closed (closedCarryAfter activeLast lastClaim)) claims after) :
    claims = [] :=
  (VerifiedRun.from_closed_is_empty run).1

theorem verified_transition_uses_the_consumed_claim : verifiesLast lastClaim :=
  verified_suffix verifiedClose

/- A raw semantic transition does not provide verifier authority. The wrapper
cannot exist when the verifier rejects that same claim. -/
namespace MissingVerification

def rejectsAll (_claim : ClaimSuffix Nat Nat Nat) : Prop := False

theorem rejected_claim_cannot_form_verified_transition :
    ¬ VerifiedTransition rejectsAll balanced (.active activeLast) lastClaim
      (.closed (closedCarryAfter activeLast lastClaim)) := by
  intro transition
  exact transition.verified

end MissingVerification

/- Step zero cannot close a segment even if a caller proposes a closed output. -/
theorem early_close_is_impossible :
    ¬ ∃ closed : ClosedCarry Nat,
      Consumes balanced (.active activeFirst) firstClaim (.closed closed) := by
  rintro ⟨closed, transition⟩
  have required := active_to_closed_requires_all_close_checks transition
  have last := required.1
  change 1 = 1088 at last
  omega

theorem closed_state_cannot_consume_another_claim :
    ¬ Consumes balanced
      (.closed (closedCarryAfter activeLast lastClaim) : Carry Nat Nat Nat)
      lastClaim
      (.closed (closedCarryAfter activeLast lastClaim) : Carry Nat Nat Nat) :=
  cannot_consume_from_closed

def deriveChallenge
    (_closed : ClosedCarry Nat) (roots : Roots Nat) (count : Nat) : Nat :=
  roots.operations + roots.initialSnapshot + roots.finalSnapshot + count

theorem open_derives_after_receiving_precommit :
    match openSegment (ChallengeField := Nat) deriveChallenge chainHeaders precommit 6
        ({ segmentIndex := 0
           globalTimestamp := 0
           memoryRoot := 20 } : ClosedCarry Nat)
        (by exact ⟨by decide, by decide⟩) (by decide) (by decide) with
    | .active active => active.challenge = 66
    | .closed _ => False := by
  rfl

theorem open_uses_one_products :
    match openSegment (ChallengeField := Nat) deriveChallenge chainHeaders precommit 6
        ({ segmentIndex := 0
           globalTimestamp := 0
           memoryRoot := 20 } : ClosedCarry Nat)
        (by exact ⟨by decide, by decide⟩) (by decide) (by decide) with
    | .active active => active.products = ProductState.one
    | .closed _ => False := by
  rfl

end tests.NebulaV2FPrime
