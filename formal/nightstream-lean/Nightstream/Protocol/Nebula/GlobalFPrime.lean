import Nightstream.Protocol.Nebula.FullClaim

/-!
Contract: one exact lifetime F-prime chain for Nebula V2.

Assurance tier: model-level.

Owns canonical segment opening, ordered consumption of mandatory verified full
claims, exact segment closure, exact cross-segment carry ownership, flattened
lifetime claim order, and the base/recursive/terminal delayed-claim view.

Does not own NIFS extraction, generated rows, recursive-size closure, state
hash collision resistance, or the deployed terminal verifier.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.GlobalFPrime

open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.FullClaim
open Nightstream.Protocol.Nebula.Lifecycle

abbrev V2Verifier
    (schema : Schema) (Digest ChallengeField : Type) :=
  schema.NifsProof →
    Claim schema Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField) → Prop

abbrev Receipt
    (schema : Schema) (Digest ChallengeField : Type)
    (verify : V2Verifier schema Digest ChallengeField) :=
  Verified schema Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField) verify

/-- One complete segment starts from a closed carry, opens with canonical
products and headers, consumes exactly one verified full claim per checked
step, and reaches the next closed carry. -/
structure SegmentRun
    {ChallengeField : Type} [Field ChallengeField]
    (schema : Schema) (Digest : Type)
    (verify : V2Verifier schema Digest ChallengeField)
    (before : ClosedCarry Digest) where
  derive :
    ClosedCarry Digest → Roots Digest → Nat →
      ProductState.Challenges ChallengeField
  headers : ChainHeaders Digest
  precommit : Roots Digest
  activeAccessCount : Nat
  canOpen : before.CanOpen
  activeCountInRange : activeAccessCount < operationCountLimit
  endTimestampInRange :
    before.globalTimestamp + activeAccessCount < timestampLimit
  active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField)
  after : ClosedCarry Digest
  claims : List (Receipt schema Digest ChallengeField verify)
  opened :
    openSegment derive headers precommit activeAccessCount before canOpen
      activeCountInRange endTimestampInRange = .active active
  consumed :
    VerifiedRun verify ProductState.Balanced
      (.active active) claims (.closed after)

namespace SegmentRun

theorem startsAtStepZero
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    run.active.stepIndex.val = 0 := by
  have activeExact := Carry.active.inj run.opened
  exact (congrArg (fun active => active.stepIndex.val) activeExact).symm

theorem startsFromExactClosedCarry
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    run.active.segmentIndex = before.segmentIndex ∧
      run.active.globalTimestamp = before.globalTimestamp ∧
      run.active.memoryRoot = before.memoryRoot ∧
      run.active.products = ProductState.one ∧
      run.active.dSeen = run.headers.roots := by
  have activeExact := Carry.active.inj run.opened
  exact
    ⟨(congrArg (fun active => active.segmentIndex) activeExact).symm,
      (congrArg (fun active => active.globalTimestamp) activeExact).symm,
      (congrArg (fun active => active.memoryRoot) activeExact).symm,
      (congrArg (fun active => active.products) activeExact).symm,
      (congrArg (fun active => active.dSeen) activeExact).symm⟩

theorem exactClaimCount
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    run.claims.length = claimsPerSegment :=
  run.consumed.full_segment_has_exact_claim_count run.startsAtStepZero

theorem afterSegmentIndex
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    run.after.segmentIndex = before.segmentIndex + 1 := by
  have activeStart : run.active.segmentIndex = before.segmentIndex :=
    (congrArg (fun active => active.segmentIndex)
      (Carry.active.inj run.opened)).symm
  calc
    run.after.segmentIndex = run.active.segmentIndex + 1 :=
      run.consumed.to_closed_segment_index
    _ = before.segmentIndex + 1 := by rw [activeStart]

theorem everyClaimAccepted
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    ∀ receipt ∈ run.claims, verify receipt.proof receipt.claim :=
  run.consumed.every_claim_accepted

/-- Segment closure exposes the exact two-repetition product balance check.
The caller cannot replace this predicate with `True` or another relation. -/
theorem finalProductsBalanced
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before) :
    ∃ receipt ∈ run.claims,
      ProductState.Balanced receipt.claim.memory.productsAfter :=
  run.consumed.to_closed_has_balanced_products

end SegmentRun

/-- A lifetime chain owns the exact closed carry passed from one segment to
the next. Its claim list is the exact ordered concatenation of all segment
runs. -/
inductive Chain
    {ChallengeField : Type} [Field ChallengeField]
    (schema : Schema) (Digest : Type)
    (verify : V2Verifier schema Digest ChallengeField) :
    ClosedCarry Digest →
      List (Receipt schema Digest ChallengeField verify) →
      ClosedCarry Digest → Nat → Prop
  | nil (state : ClosedCarry Digest) :
      Chain schema Digest verify state [] state 0
  | cons
      {before final : ClosedCarry Digest}
      {tailClaims : List (Receipt schema Digest ChallengeField verify)}
      {tailSegments : Nat}
      (head : SegmentRun schema Digest verify before)
      (tail : Chain schema Digest verify head.after tailClaims final tailSegments) :
      Chain schema Digest verify before (head.claims ++ tailClaims) final
        (tailSegments + 1)

namespace Chain

theorem exactClaimCount
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain schema Digest verify initial claims final segmentCount) :
    claims.length = segmentCount * claimsPerSegment := by
  induction chain with
  | nil => simp
  | cons head _ inductionHypothesis =>
      rw [List.length_append, head.exactClaimCount, inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

theorem finalSegmentIndex
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain schema Digest verify initial claims final segmentCount) :
    final.segmentIndex = initial.segmentIndex + segmentCount := by
  induction chain with
  | nil => rfl
  | cons head _ inductionHypothesis =>
      rw [inductionHypothesis, head.afterSegmentIndex]
      omega

theorem everyClaimAccepted
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain schema Digest verify initial claims final segmentCount) :
    ∀ receipt ∈ claims, verify receipt.proof receipt.claim := by
  induction chain with
  | nil => simp
  | cons head _ inductionHypothesis =>
      intro receipt member
      rw [List.mem_append] at member
      rcases member with headMember | tailMember
      · exact head.everyClaimAccepted receipt headMember
      · exact inductionHypothesis receipt tailMember

theorem completeDelayedSchedule
    {ChallengeField : Type} [Field ChallengeField]
    {schema : Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain schema Digest verify initial claims final segmentCount)
    (positiveSegments : 0 < segmentCount) :
    CompleteSchedule claims.length := by
  apply completeSchedule
  rw [chain.exactClaimCount]
  have stepsPositive : 0 < claimsPerSegment := by decide
  exact Nat.mul_pos positiveSegments stepsPositive

end Chain

/-- Exact full claim produced at one augmented invocation. -/
def producedClaimAt
    {ClaimType : Type} (claims : List ClaimType)
    (invocation : InvocationIndex claims.length) : Option ClaimType :=
  (producedAt invocation).map claims.get

/-- Exact full claim consumed at one augmented invocation. -/
def consumedClaimAt
    {ClaimType : Type} (claims : List ClaimType)
    (invocation : InvocationIndex claims.length) : Option ClaimType :=
  (consumedAt invocation).map claims.get

theorem base_produces_exact_first_claim
    {ClaimType : Type} {claims : List ClaimType}
    (positive : 0 < claims.length) :
    producedClaimAt claims (baseIndex claims.length) =
      some (claims.get ⟨0, positive⟩) := by
  simp [producedClaimAt, base_produces_first positive]

theorem recursive_consumes_and_produces_exact_claims
    {ClaimType : Type} (claims : List ClaimType)
    (invocation : InvocationIndex claims.length)
    (afterBase : 0 < invocation.val)
    (beforeTerminal : invocation.val < claims.length) :
    consumedClaimAt claims invocation =
        some (claims.get ⟨invocation.val - 1, by omega⟩) ∧
      producedClaimAt claims invocation =
        some (claims.get ⟨invocation.val, beforeTerminal⟩) := by
  have indexes := recursive_consumes_prior_and_produces_current invocation
    afterBase beforeTerminal
  simp [consumedClaimAt, producedClaimAt, indexes.1, indexes.2]

theorem terminal_consumes_exact_trailing_claim
    {ClaimType : Type} {claims : List ClaimType}
    (positive : 0 < claims.length) :
    consumedClaimAt claims (terminalIndex claims.length) =
      some (claims.get ⟨claims.length - 1, by omega⟩) := by
  simp [consumedClaimAt, terminal_consumes_trailing positive]

theorem terminal_produces_no_claim
    {ClaimType : Type} (claims : List ClaimType) :
    producedClaimAt claims (terminalIndex claims.length) = none := by
  simp [producedClaimAt, terminal_produces_none]

end Nightstream.Protocol.Nebula.GlobalFPrime
