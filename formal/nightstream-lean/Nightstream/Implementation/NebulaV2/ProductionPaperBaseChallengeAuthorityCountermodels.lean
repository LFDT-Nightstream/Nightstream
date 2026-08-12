import Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor

/-!
Contract: hostile model for an unbound base memory-challenge authority.

The local segment-open transition is intentionally parametric in its seven
authority digests.  These examples prove that local opening correctness alone
accepts two different authorities while every other opening input is equal.
Therefore the base F-prime relation must also prove the exact authority link
owned by `ProductionPaperBaseInvocationFor.challengeAuthority`.

This is not an attack on the exact lifetime model.  Its base evidence now
requires that link.  The generated base rows must still implement it.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperBaseChallengeAuthorityCountermodels

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime

def zeroDigest : Digest.Value :=
  { lanes := fun _ => ⟨0, by decide⟩ }

def oneDigest : Digest.Value :=
  { lanes := fun _ => ⟨1, by decide⟩ }

def zeroAuthority : MemoryOpenSegment.Authority :=
  { verifierKeyDigest := zeroDigest
    applicationRelationDigest := zeroDigest
    programDigest := zeroDigest
    memoryPlanDigest := zeroDigest
    laneLayoutDigest := zeroDigest
    priorStateDigest := zeroDigest
    runningAccumulatorDigest := zeroDigest }

def changedVerifierKeyAuthority : MemoryOpenSegment.Authority :=
  { zeroAuthority with verifierKeyDigest := oneDigest }

theorem zeroDigest_ne_oneDigest : zeroDigest ≠ oneDigest := by
  intro equal
  have lane := congrArg (fun digest => (digest.lanes (0 : Fin 4)).val) equal
  norm_num [zeroDigest, oneDigest] at lane

theorem authorities_differ :
    zeroAuthority ≠ changedVerifierKeyAuthority := by
  intro equal
  have digestEqual := congrArg MemoryOpenSegment.Authority.verifierKeyDigest equal
  exact zeroDigest_ne_oneDigest digestEqual

def headers : ChainHeaders Digest.Value :=
  { operations := zeroDigest
    memory := zeroDigest }

def roots : Roots Digest.Value :=
  { operations := zeroDigest
    initialSnapshot := zeroDigest
    finalSnapshot := zeroDigest }

/-- The countermodel uses one exact field-native profile. It does not rely on
the fixed-V2 reference wrappers. -/
def profileCandidate : ProductionProfileCandidates.Id := .e1

def opening (authority : MemoryOpenSegment.Authority) :
    ProductionPaperBaseInvocationFor.Opening :=
  { initialMemoryRoot := zeroDigest
    authority := authority
    precommit := roots
    activeAccessCount := 0
    activeCountInRange := by decide
    endTimestampInRange := by decide }

/-- Local opening correctness does not select the base authority. -/
def LocallyExact
    (openingCandidate : ProductionPaperBaseInvocationFor.Opening) : Prop :=
  MemoryOpenSegment.openCarryFor
      (ProductionProfileCandidates.identity profileCandidate)
      openingCandidate.authority headers
      openingCandidate.precommit openingCandidate.activeAccessCount
      (ProductionPaperBaseInvocationFor.initialClosed
        openingCandidate.initialMemoryRoot)
      (ProductionPaperBaseInvocationFor.initialClosed_canOpen
        openingCandidate.initialMemoryRoot)
      openingCandidate.activeCountInRange
      openingCandidate.initialEndTimestampInRange =
    .active (openingCandidate.activeFor profileCandidate headers)

theorem every_authority_is_locally_exact
    (authority : MemoryOpenSegment.Authority) :
    LocallyExact (opening authority) :=
  (opening authority).open_exact_for profileCandidate headers

/-- Two different authorities satisfy the same local opening interface while
the root, precommitment, and active count remain equal.  The missing
authority-equality condition is therefore necessary, not documentary. -/
theorem local_opening_does_not_bind_base_authority :
    ∃ left right : ProductionPaperBaseInvocationFor.Opening,
      left.authority ≠ right.authority ∧
      left.initialMemoryRoot = right.initialMemoryRoot ∧
      left.precommit = right.precommit ∧
      left.activeAccessCount = right.activeAccessCount ∧
      LocallyExact left ∧ LocallyExact right := by
  refine ⟨opening zeroAuthority, opening changedVerifierKeyAuthority,
    authorities_differ, rfl, rfl, rfl, ?_, ?_⟩
  · exact every_authority_is_locally_exact zeroAuthority
  · exact every_authority_is_locally_exact changedVerifierKeyAuthority

end Nightstream.Implementation.NebulaV2.ProductionPaperBaseChallengeAuthorityCountermodels
