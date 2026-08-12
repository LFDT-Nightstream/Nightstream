import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation
import Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage
import Nightstream.Protocol.NebulaV2.IdealFingerprint

/-!
Contract: derive sequential memory execution from one complete row-derived
production segment.

The theorem reconstructs the two canonical snapshots, the ordered active
access list, all four record multisets, and both fingerprint repetitions from
the checked rows. It concludes exact sequential execution or the named event
that a nonzero concrete fingerprint difference evaluates to zero at both
transcript challenge pairs.

No premise states coverage, product acceptance, multiset balance, or memory
execution.

Does not own challenge unpredictability, root binding, application-control
rows, NIFS extraction, recursive-size closure, or deployed-verifier
refinement.

Assurance tier: implementation-to-protocol bridge with an explicit
cryptographic bad-event boundary.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation
open Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments
open Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage
open Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

/-- The field structure transported through the proved coefficient
equivalence uses the same concrete multiplicative identity as the row layer. -/
noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

namespace SegmentRun

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}
variable {before : ClosedCarry Digest.Value}

/-- The exact operation list is fixed by the physical-order row sources in
the segment's checked steps. -/
def accesses
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) : List Access :=
  ProductionMemoryStepSemantics.Run.accesses (steps run.batches)

/-- The row-derived opening-to-close schedule is one strict global integer
timestamp schedule. -/
theorem orderedActiveToClosed
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    Ordered run.active.globalTimestamp (accesses run)
      run.after.globalTimestamp := by
  simpa [accesses, carryTimestamp] using run.consumed.toStepRun.ordered

/-- The row-derived segment covers both complete snapshots and every
application operation exactly once, with multiset multiplicity preserved. -/
theorem covers
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    ProductState.Covers
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .initialSnapshot)
      (accesses run)
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .finalSnapshot)
      (ProductionMemoryStepSemantics.Run.chunks (steps run.batches)) where
  initialSnapshot := by
    simpa [ProductionMemorySnapshotCoverage.chunkSnapshot] using
      ProductionMemorySnapshotCoverage.SegmentRun.snapshotChunksCover run
        .initialSnapshot
  writes := by
    simpa [accesses] using
      ProductionMemoryStepSemantics.Run.writesCover (steps run.batches)
  reads := by
    simpa [accesses] using
      ProductionMemoryStepSemantics.Run.readsCover (steps run.batches)
  finalSnapshot := by
    simpa [ProductionMemorySnapshotCoverage.chunkSnapshot] using
      ProductionMemorySnapshotCoverage.SegmentRun.snapshotChunksCover run
        .finalSnapshot

/-- Exact ideal fingerprint check represented by the complete row-derived
segment. Record bounds are conclusions of the source rows and carry rules. -/
def fingerprintCheck
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    IdealFingerprint.Check encode
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .initialSnapshot)
      (accesses run)
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .finalSnapshot) where
  bounds := by
    have wellFormed :=
      ProductionMemoryProductAccumulation.Run.activeWellFormed
        run.consumed.toStepRun
    have startInRange : run.active.segmentStartTimestamp < timestampLimit :=
      (wellFormed.2.2.2.2.1.trans wellFormed.2.2.2.2.2).trans_lt
        wellFormed.2.2.2.1
    exact RecordBounds.ofValidAt
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshotValidAt run
        .initialSnapshot)
      startInRange
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshotValidAt run
        .finalSnapshot)
      wellFormed.2.2.2.1
      (orderedActiveToClosed run)
  challenges := mapChallenges run.active.challenge

private theorem fieldOneProducts_eq_concreteOne :
    (ProductState.one : State K) = MemoryCarryCodec.oneProductsK := by
  funext repetition
  apply ProductState.Four.ext <;>
    apply ConcreteField.superNeoEquiv.injective <;>
    simp only [ProductState.one, MemoryCarryCodec.oneProductsK,
      ConcreteField.superNeoEquiv_one]
  all_goals
    change ConcreteField.superNeoEquiv
      (ConcreteField.superNeoEquiv.symm (1 : ChallengeField)) = 1
    exact ConcreteField.superNeoEquiv.apply_symm_apply 1

/-- Canonical segment opening fixes all eight concrete products to one. This
is derived from `openSegment`; callers do not supply it. -/
theorem openingProductsConcrete
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    run.active.products = MemoryCarryCodec.oneProductsK := by
  calc
    run.active.products = ProductState.one :=
      run.toProtocol.startsFromExactClosedCarry.2.2.2.1
    _ = MemoryCarryCodec.oneProductsK := fieldOneProducts_eq_concreteOne

/-- Closing product rows and exact row-derived coverage force both concrete
fingerprint repetitions to accept. -/
theorem fingerprintAccepted
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    (fingerprintCheck run).Accepts := by
  have accumulated :=
    ProductionMemoryProductAccumulation.Run.accumulatedFromConcreteOneBalanced
      run.consumed.toStepRun (openingProductsConcrete run)
  have coverage := covers run
  have expectedBalanced :
      ProductState.Balanced
        (ProductState.expected (fingerprintCheck run)) := by
    rw [← ProductState.accumulate_one_eq_expected
      (fingerprintCheck run) coverage]
    simpa [fingerprintCheck] using accumulated
  exact (ProductState.accepts_iff_expected_balanced
    (fingerprintCheck run)).mpr expectedBalanced

/-- Exact multiset balance follows unless the concrete nonzero difference
polynomial evaluates to zero at both transcript challenge pairs. -/
theorem balanceOrEvaluationFailure
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    Memory.Balanced
        (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
          .initialSnapshot).tuples
        (accesses run)
        (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
          .finalSnapshot).tuples ∨
      IdealFingerprint.EvaluationFailure (fingerprintCheck run) := by
  exact IdealFingerprint.balance_or_evaluationFailure
    ConcreteField.encode_injective_below_goldilocks
    (fingerprintCheck run) (fingerprintAccepted run)

/-- Central non-circular segment theorem. Satisfying production rows derive
the exact sequential memory execution, except for the named concrete
fingerprint evaluation event. -/
theorem executesOrEvaluationFailure
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    Memory.Executes
        (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
          .initialSnapshot).tuples
        run.active.globalTimestamp
        (accesses run)
        (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
          .finalSnapshot).tuples
        run.after.globalTimestamp ∨
      IdealFingerprint.EvaluationFailure (fingerprintCheck run) := by
  rcases balanceOrEvaluationFailure run with balance | failure
  · exact Or.inl (Memory.balanced_implies_executes
      (orderedActiveToClosed run) balance)
  · exact Or.inr failure

end SegmentRun

end Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness
