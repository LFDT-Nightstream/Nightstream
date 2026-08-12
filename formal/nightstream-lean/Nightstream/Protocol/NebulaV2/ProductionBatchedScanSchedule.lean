import Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

/-!
Contract: exact flattened checked-step order for production batch profiles.

One verified production claim contains `E` ordered memory suffixes. This file
forgets only that outer grouping and proves that the resulting suffix list has
the same strict step-index and segment-boundary schedule as the independent
factor-one F-prime model.

No claim count, step-index list, or segment timestamp list is an input to the
ordering theorems. They follow from the exact `ConsumesList` transitions.

Does not own snapshot values, generated rows, challenge security, or proof
extraction.

Assurance tier: independent protocol model.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule

open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

namespace ConsumesList

/-- Every suffix position has the exact step index forced by its preceding
carry. This theorem uses the transition chain, not list length. -/
theorem claim_step_at
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : ConsumesList balanced before claims after)
    (active : ActiveCarry Digest Challenge Products)
    (beforeActive : before = .active active)
    (index : Fin claims.length) :
    (claims.get index).stepIndex.val =
      active.stepIndex.val + index.val := by
  induction run generalizing active with
  | nil =>
      exact Fin.elim0 index
  | @cons before middle after head tail step rest inductionHypothesis =>
      cases step with
      | @interior activeBefore claim agreement notLast =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · simpa using congrArg Fin.val agreement.stepIndex
          · have tailStep := inductionHypothesis
              (interiorCarry activeBefore head notLast) rfl tailIndex
            change
              (tail.get tailIndex).stepIndex.val =
                activeBefore.stepIndex.val + tailIndex.succ.val
            simp only [interiorCarry] at tailStep
            have successorValue : tailIndex.succ.val = tailIndex.val + 1 :=
              rfl
            rw [successorValue]
            omega
      | @close activeBefore claim agreement last checks =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          have tailEmpty := rest.from_closed_is_empty
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · simpa using congrArg Fin.val agreement.stepIndex
          · have noTail : tail.length = 0 := by simp [tailEmpty.1]
            have tailBound := tailIndex.isLt
            omega

/-- Every suffix carries the exact segment timestamp bounds from the opening
active carry. The proof follows the transition chain across all inner batch
positions. -/
theorem claim_segment_bounds_at
    {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List (ClaimSuffix Digest Challenge Products)}
    (run : ConsumesList balanced before claims after)
    (active : ActiveCarry Digest Challenge Products)
    (beforeActive : before = .active active)
    (index : Fin claims.length) :
    (claims.get index).segmentStartTimestamp =
        active.segmentStartTimestamp /\
      (claims.get index).segmentEndTimestamp =
        active.segmentEndTimestamp := by
  induction run generalizing active with
  | nil =>
      exact Fin.elim0 index
  | @cons before middle after head tail step rest inductionHypothesis =>
      cases step with
      | @interior activeBefore claim agreement notLast =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · exact ⟨agreement.segmentStartTimestamp,
              agreement.segmentEndTimestamp⟩
          · have tailBounds := inductionHypothesis
              (interiorCarry activeBefore head notLast) rfl tailIndex
            simpa [interiorCarry] using tailBounds
      | @close activeBefore claim agreement last checks =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          have tailEmpty := rest.from_closed_is_empty
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · exact ⟨agreement.segmentStartTimestamp,
              agreement.segmentEndTimestamp⟩
          · have noTail : tail.length = 0 := by simp [tailEmpty.1]
            have tailBound := tailIndex.isLt
            omega

end ConsumesList

namespace SegmentRun

/-- The complete flattened suffix list for one production segment. -/
def suffixes
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : ProductionProfileCandidates.Id}
    {schema : ProductionBatchedFPrime.Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : ProductionBatchedGlobalFPrime.SegmentRun candidate schema Digest
      verify derive headers before) :
    List (ClaimSuffix Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)) :=
  run.claims.flatMap fun receipt => receipt.claim.memory.suffixes

theorem suffixes_length_exact
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : ProductionProfileCandidates.Id}
    {schema : ProductionBatchedFPrime.Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : ProductionBatchedGlobalFPrime.SegmentRun candidate schema Digest
      verify derive headers before) :
    (suffixes run).length = Lifecycle.claimsPerSegment := by
  have exact := run.exactSuffixCount
  simpa [SegmentRun.suffixes,
    ProductionBatchedFPrime.VerifiedRun.totalSuffixCount,
    ProductionProfileCandidates.stepsPerSegment] using exact

/-- A complete production segment fixes every flattened step index to its
exact list position `0, ..., 1087`. -/
theorem suffix_step_at
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : ProductionProfileCandidates.Id}
    {schema : ProductionBatchedFPrime.Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : ProductionBatchedGlobalFPrime.SegmentRun candidate schema Digest
      verify derive headers before)
    (index : Fin (suffixes run).length) :
    ((suffixes run).get index).stepIndex.val = index.val := by
  have indexed := ConsumesList.claim_step_at
    run.consumed.flattenConsumes run.active rfl index
  rw [run.startsAtStepZero] at indexed
  simpa only [suffixes, Nat.zero_add] using indexed

/-- Every flattened suffix uses the one segment start and end timestamp fixed
at the segment opening. -/
theorem suffix_segment_bounds_at
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : ProductionProfileCandidates.Id}
    {schema : ProductionBatchedFPrime.Schema} {Digest : Type}
    {verify : BatchVerifier candidate schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {before : ClosedCarry Digest}
    (run : ProductionBatchedGlobalFPrime.SegmentRun candidate schema Digest
      verify derive headers before)
    (index : Fin (suffixes run).length) :
    ((suffixes run).get index).segmentStartTimestamp =
        run.active.segmentStartTimestamp /\
      ((suffixes run).get index).segmentEndTimestamp =
        run.active.segmentEndTimestamp := by
  simpa only [suffixes] using
    (ConsumesList.claim_segment_bounds_at run.consumed.flattenConsumes
      run.active rfl index)

end SegmentRun

end Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule
