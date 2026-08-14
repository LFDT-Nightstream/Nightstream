import Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle

/-!
Contract: reverse the batch-aware delayed F-prime schedule into exact
segments.

The forward compiler theorem already maps a positive segment chain to the
base-opened delayed schedule. This file proves the converse. Starting from
the exact base opening, it partitions every delayed transition at the unique
closed-carry continuations and reconstructs the complete segment chain.

The result closes a lifecycle theorem gap. It does not prove generated-row
refinement, NIFS extraction, challenge security, or terminal cryptography.

Assurance tier: independent protocol model.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse

open Nightstream.Protocol.Nebula.AugmentedLifecycle
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

namespace VerifiedRun

/-- Exact verified batch runs compose without changing claim order. -/
theorem append
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : ProductionBatchedGlobalFPrime.BatchVerifier candidate schema
      Digest ChallengeField}
    {before middle after : Carry Digest
      (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {left right : List
      (ProductionBatchedGlobalFPrime.Receipt candidate schema Digest
        ChallengeField verify)}
    (first : ProductionBatchedFPrime.VerifiedRun verify
      ProductState.Balanced before left middle)
    (second : ProductionBatchedFPrime.VerifiedRun verify
      ProductState.Balanced middle right after) :
    ProductionBatchedFPrime.VerifiedRun verify ProductState.Balanced before
      (left ++ right) after := by
  induction first with
  | nil => exact second
  | cons step _ inductionHypothesis =>
      exact .cons step (inductionHypothesis second)

end VerifiedRun

/-- Complete a partially accumulated current segment and every delayed tail.
The positive segment count is derived because a delayed run always contains
its trailing terminal claim. -/
private theorem completeFromPrefix
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : ProductionBatchedGlobalFPrime.BatchVerifier candidate schema
      Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {segmentBefore final : ClosedCarry Digest}
    {active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {current : Carry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {prefixClaims tailClaims : List
      (ProductionBatchedGlobalFPrime.Receipt candidate schema Digest
        ChallengeField verify)}
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (canOpen : segmentBefore.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      segmentBefore.globalTimestamp + activeAccessCount < timestampLimit)
    (opened : openSegment derive headers precommit activeAccessCount
      segmentBefore canOpen activeCountInRange endTimestampInRange =
        .active active)
    (prefixRun : ProductionBatchedFPrime.VerifiedRun verify
      ProductState.Balanced (.active active) prefixClaims current)
    (delayed : DelayedRun verify derive headers current tailClaims final) :
    exists segmentCount,
      ProductionBatchedGlobalFPrime.Chain candidate schema Digest verify derive
          headers segmentBefore (prefixClaims ++ tailClaims) final
            segmentCount /\
        0 < segmentCount := by
  induction delayed generalizing segmentBefore active prefixClaims precommit
      activeAccessCount canOpen activeCountInRange endTimestampInRange with
  | @terminal current receipt terminalFinal consumes =>
      let completeRun : ProductionBatchedFPrime.VerifiedRun verify
          ProductState.Balanced (.active active)
          (prefixClaims ++ [receipt]) (.closed terminalFinal) :=
        VerifiedRun.append prefixRun
          (ProductionBatchedFPrime.VerifiedRun.cons consumes
            (ProductionBatchedFPrime.VerifiedRun.nil _))
      let head : ProductionBatchedGlobalFPrime.SegmentRun candidate schema
          Digest verify derive headers segmentBefore :=
        { precommit := precommit
          activeAccessCount := activeAccessCount
          canOpen := canOpen
          activeCountInRange := activeCountInRange
          endTimestampInRange := endTimestampInRange
          active := active
          after := terminalFinal
          claims := prefixClaims ++ [receipt]
          opened := opened
          consumed := completeRun }
      refine ⟨1, ?_, by omega⟩
      simpa [head] using
        (ProductionBatchedGlobalFPrime.Chain.cons head
          (ProductionBatchedGlobalFPrime.Chain.nil terminalFinal))
  | @recursive current intermediate outgoing restClaims terminalFinal receipt
      consumes continues rest inductionHypothesis =>
      let extendedRun : ProductionBatchedFPrime.VerifiedRun verify
          ProductState.Balanced (.active active)
          (prefixClaims ++ [receipt]) intermediate :=
        VerifiedRun.append prefixRun
          (ProductionBatchedFPrime.VerifiedRun.cons consumes
            (ProductionBatchedFPrime.VerifiedRun.nil _))
      cases continues with
      | interior middleActive =>
          simpa [List.append_assoc] using
            (inductionHypothesis precommit activeAccessCount canOpen
              activeCountInRange endTimestampInRange opened extendedRun)
      | boundary closed nextPrecommit nextActiveAccessCount nextCanOpen
          nextActiveCountInRange nextEndTimestampInRange =>
          let nextActive : ActiveCarry Digest
              (ProductState.Challenges ChallengeField)
              (ProductState.State ChallengeField) :=
            { segmentIndex := closed.segmentIndex
              stepIndex := ⟨0, by decide⟩
              globalTimestamp := closed.globalTimestamp
              segmentStartTimestamp := closed.globalTimestamp
              segmentActiveAccessCount := nextActiveAccessCount
              segmentEndTimestamp :=
                closed.globalTimestamp + nextActiveAccessCount
              challenge := derive closed nextPrecommit nextActiveAccessCount
              products := ProductState.one
              dPre := nextPrecommit
              dSeen := headers.roots
              memoryRoot := closed.memoryRoot }
          have nextOpened : openSegment derive headers nextPrecommit
              nextActiveAccessCount closed nextCanOpen
              nextActiveCountInRange nextEndTimestampInRange =
                .active nextActive := rfl
          have nextPrefix : ProductionBatchedFPrime.VerifiedRun verify
              ProductState.Balanced (.active nextActive) []
              (.active nextActive) :=
            ProductionBatchedFPrime.VerifiedRun.nil _
          rcases inductionHypothesis nextPrecommit nextActiveAccessCount
              nextCanOpen nextActiveCountInRange nextEndTimestampInRange
              nextOpened nextPrefix with
            ⟨tailSegments, tailChain, tailPositive⟩
          let head : ProductionBatchedGlobalFPrime.SegmentRun candidate schema
              Digest verify derive headers segmentBefore :=
            { precommit := precommit
              activeAccessCount := activeAccessCount
              canOpen := canOpen
              activeCountInRange := activeCountInRange
              endTimestampInRange := endTimestampInRange
              active := active
              after := closed
              claims := prefixClaims ++ [receipt]
              opened := opened
              consumed := extendedRun }
          refine ⟨tailSegments + 1, ?_, by omega⟩
          simpa [head, List.append_assoc] using
            (ProductionBatchedGlobalFPrime.Chain.cons head tailChain)

/-- Reverse compiler theorem for the exact base-opened delayed schedule.
No segment partition, claim count, or boundary list is supplied by the
caller. -/
theorem delayedRun_to_segmentChain
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : ProductionBatchedGlobalFPrime.BatchVerifier candidate schema
      Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List
      (ProductionBatchedGlobalFPrime.Receipt candidate schema Digest
        ChallengeField verify)}
    (precommit : Roots Digest)
    (activeAccessCount : Nat)
    (canOpen : initial.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      initial.globalTimestamp + activeAccessCount < timestampLimit)
    (opened : openSegment derive headers precommit activeAccessCount initial
      canOpen activeCountInRange endTimestampInRange = .active active)
    (delayed : DelayedRun verify derive headers (.active active) claims final) :
    exists segmentCount,
      ProductionBatchedGlobalFPrime.Chain candidate schema Digest verify derive
          headers initial claims final segmentCount /\
        0 < segmentCount := by
  simpa using completeFromPrefix precommit activeAccessCount canOpen
    activeCountInRange endTimestampInRange opened
    (ProductionBatchedFPrime.VerifiedRun.nil (.active active)) delayed

/-- Forward and reverse lifecycle constructions agree at the theorem level:
every positive exact segment chain has a delayed schedule, and every exact
base-opened delayed schedule has a positive segment chain. -/
theorem segmentChain_iff_delayedRun
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest : Type}
    {verify : ProductionBatchedGlobalFPrime.BatchVerifier candidate schema
      Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {initial final : ClosedCarry Digest}
    {claims : List
      (ProductionBatchedGlobalFPrime.Receipt candidate schema Digest
        ChallengeField verify)} :
    (exists segmentCount,
      ProductionBatchedGlobalFPrime.Chain candidate schema Digest verify derive
          headers initial claims final segmentCount /\
        0 < segmentCount) <->
      (exists precommit activeAccessCount canOpen activeCountInRange
          endTimestampInRange active,
        openSegment derive headers precommit activeAccessCount initial canOpen
            activeCountInRange endTimestampInRange = .active active /\
          DelayedRun verify derive headers (.active active) claims final) := by
  constructor
  · rintro ⟨segmentCount, chain, positive⟩
    exact SegmentChain.toDelayedRun chain positive
  · rintro ⟨precommit, activeAccessCount, canOpen, activeCountInRange,
      endTimestampInRange, active, opened, delayed⟩
    exact delayedRun_to_segmentChain precommit activeAccessCount canOpen
      activeCountInRange endTimestampInRange opened delayed

end Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse
