import Nightstream.Protocol.NebulaV2.GlobalFPrime
import Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime

/-!
Contract: exact lifetime F-prime chain for a field-native batch candidate.

Each segment opens from the exact prior closed carry, consumes the verified
candidate-specific batch claims in order, and reaches the next closed carry.
The lifetime claim list is the exact ordered concatenation of its segments.
The generic delayed base, recursive, and terminal indexes are instantiated
with the candidate-specific claim count.

Does not own generated rows, proof extraction, challenge security, terminal
backend implementation, or candidate selection.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

abbrev BatchVerifier
    (candidate : Id) (schema : Schema)
    (Digest ChallengeField : Type) :=
  Verifier candidate schema Digest
    (ProductState.Challenges ChallengeField) (ProductState.State ChallengeField)

abbrev Receipt
    (candidate : Id) (schema : Schema)
    (Digest ChallengeField : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField) :=
  Verified candidate schema Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField) verify

/-- One complete candidate-specific segment. -/
structure SegmentRun
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (derive :
      ClosedCarry Digest -> Roots Digest -> Nat ->
        ProductState.Challenges ChallengeField)
    (headers : ChainHeaders Digest)
    (before : ClosedCarry Digest) where
  precommit : Roots Digest
  activeAccessCount : Nat
  canOpen : before.CanOpen
  activeCountInRange : activeAccessCount < operationCountLimit
  endTimestampInRange :
    before.globalTimestamp + activeAccessCount < timestampLimit
  active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField)
  after : ClosedCarry Digest
  claims : List (Receipt candidate schema Digest ChallengeField verify)
  opened :
    openSegment derive headers precommit activeAccessCount before canOpen
      activeCountInRange endTimestampInRange = .active active
  consumed :
    ProductionBatchedFPrime.VerifiedRun verify ProductState.Balanced
      (.active active) claims (.closed after)

namespace SegmentRun

theorem startsAtStepZero
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    run.active.stepIndex.val = 0 := by
  have activeExact := Carry.active.inj run.opened
  exact (congrArg (fun active => active.stepIndex.val) activeExact).symm

theorem startsFromExactClosedCarry
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    run.active.segmentIndex = before.segmentIndex /\
      run.active.globalTimestamp = before.globalTimestamp /\
      run.active.memoryRoot = before.memoryRoot /\
      run.active.products = ProductState.one /\
      run.active.dSeen = headers.roots := by
  have activeExact := Carry.active.inj run.opened
  exact
    ⟨(congrArg (fun active => active.segmentIndex) activeExact).symm,
      (congrArg (fun active => active.globalTimestamp) activeExact).symm,
      (congrArg (fun active => active.memoryRoot) activeExact).symm,
      (congrArg (fun active => active.products) activeExact).symm,
      (congrArg (fun active => active.dSeen) activeExact).symm⟩

theorem exactClaimCount
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    run.claims.length = claimsPerSegment candidate :=
  run.consumed.full_segment_has_exact_batch_count run.startsAtStepZero

theorem exactSuffixCount
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    ProductionBatchedFPrime.VerifiedRun.totalSuffixCount run.claims =
      ProductionProfileCandidates.stepsPerSegment := by
  rw [ProductionBatchedFPrime.VerifiedRun.totalSuffixCount_exact,
    run.exactClaimCount, exact_segment_partition]

theorem afterSegmentIndex
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    run.after.segmentIndex = before.segmentIndex + 1 := by
  have activeStart : run.active.segmentIndex = before.segmentIndex :=
    run.startsFromExactClosedCarry.1
  calc
    run.after.segmentIndex = run.active.segmentIndex + 1 :=
      run.consumed.flattenConsumes.to_closed_segment_index
    _ = before.segmentIndex + 1 := by rw [activeStart]

theorem everyClaimAccepted
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    ∀ receipt ∈ run.claims, verify receipt.proof receipt.claim :=
  run.consumed.every_claim_accepted

theorem finalProductsBalanced
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : SegmentRun candidate schema Digest verify derive headers before) :
    ∃ suffix ∈
        (run.claims.flatMap fun receipt =>
          receipt.claim.memory.suffixes),
      ProductState.Balanced suffix.productsAfter :=
  run.consumed.flattenConsumes.to_closed_has_balanced_products

end SegmentRun

/-- Exact cross-segment closed-carry chain. -/
inductive Chain
    {ChallengeField : Type} [Field ChallengeField]
    (candidate : Id) (schema : Schema) (Digest : Type)
    (verify : BatchVerifier candidate schema Digest ChallengeField)
    (derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField)
    (headers : ChainHeaders Digest) :
    ClosedCarry Digest ->
      List (Receipt candidate schema Digest ChallengeField verify) ->
      ClosedCarry Digest -> Nat -> Prop
  | nil (state : ClosedCarry Digest) :
      Chain candidate schema Digest verify derive headers state [] state 0
  | cons
      {before final : ClosedCarry Digest}
      {tailClaims : List
        (Receipt candidate schema Digest ChallengeField verify)}
      {tailSegments : Nat}
      (head : SegmentRun candidate schema Digest verify derive headers before)
      (tail : Chain candidate schema Digest verify derive headers head.after
        tailClaims final tailSegments) :
      Chain candidate schema Digest verify derive headers before
        (head.claims ++ tailClaims) final (tailSegments + 1)

namespace Chain

theorem exactClaimCount
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain candidate schema Digest verify derive headers initial claims final
      segmentCount) :
    claims.length = segmentCount * claimsPerSegment candidate := by
  induction chain with
  | nil => simp
  | cons head _ inductionHypothesis =>
      rw [List.length_append, head.exactClaimCount, inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

theorem exactSuffixCount
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain candidate schema Digest verify derive headers initial claims final
      segmentCount) :
    ProductionBatchedFPrime.VerifiedRun.totalSuffixCount claims =
      segmentCount * ProductionProfileCandidates.stepsPerSegment := by
  rw [ProductionBatchedFPrime.VerifiedRun.totalSuffixCount_exact,
    chain.exactClaimCount]
  cases candidate <;>
    simp [ProductionProfileCandidates.claimsPerSegment,
      checkedStepsPerFreshClaim, ProductionProfileCandidates.stepsPerSegment,
      Nat.mul_assoc]

theorem finalSegmentIndex
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain candidate schema Digest verify derive headers initial claims final
      segmentCount) :
    final.segmentIndex = initial.segmentIndex + segmentCount := by
  induction chain with
  | nil => rfl
  | cons head _ inductionHypothesis =>
      rw [inductionHypothesis, head.afterSegmentIndex]
      omega

theorem everyClaimAccepted
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain candidate schema Digest verify derive headers initial claims final
      segmentCount) :
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
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List (Receipt candidate schema Digest ChallengeField verify)}
    {segmentCount : Nat}
    (chain : Chain candidate schema Digest verify derive headers initial claims final
      segmentCount)
    (positiveSegments : 0 < segmentCount) :
    CompleteSchedule claims.length := by
  apply completeSchedule
  rw [chain.exactClaimCount]
  have claimsPositive : 0 < claimsPerSegment candidate := by
    cases candidate <;> decide
  exact Nat.mul_pos positiveSegments claimsPositive

end Chain

end Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime
