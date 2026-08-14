import Nightstream.Implementation.Nebula.Production.Memory.SegmentSoundness
import Nightstream.Protocol.Nebula.Chain
import Nightstream.Protocol.Nebula.IdealAcceptance

/-!
Contract: compose the exact row-derived production segments into one memory
execution.

Each segment supplies its reconstructed snapshots and ordered access list.
The theorem derives sequential execution for each segment, aligns that access
list with the producer-batch application ports, checks both snapshot roots,
and joins adjacent snapshots. It returns a named fingerprint event, snapshot
authority failure, or snapshot-root collision instead of assuming any of
those conditions away.

No premise states multiset balance, segment execution, snapshot equality, or
global execution.

This file does not own the probability of a fingerprint event, snapshot-root
collision resistance, commitment-chain binding, generated-row extraction,
or deployed-verifier refinement.

Assurance tier: implementation-to-protocol bridge with explicit bad events.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionMemoryChainSoundness

open Nightstream.Implementation.Nebula.ProductionMemoryRowSegments
open Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage
open Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

/-- Every deterministic obstruction to composing row-derived memory
segments. -/
inductive Failure
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value)
    (snapshotRoot : Snapshot -> Digest.Value) : Prop where
  | fingerprint
      {before : ClosedCarry Digest.Value}
      (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
        headers before)
      (failure : IdealFingerprint.EvaluationFailure
        (ProductionMemorySegmentSoundness.SegmentRun.fingerprintCheck run)) :
      Failure verify headers snapshotRoot
  | initialRoot
      {before : ClosedCarry Digest.Value}
      (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
        headers before)
      (mismatch : snapshotRoot
          (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
            .initialSnapshot) ≠ before.memoryRoot) :
      Failure verify headers snapshotRoot
  | finalRoot
      {before : ClosedCarry Digest.Value}
      (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
        headers before)
      (mismatch : snapshotRoot
          (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
            .finalSnapshot) ≠ run.after.memoryRoot) :
      Failure verify headers snapshotRoot
  | snapshotCollision
      (collision : IdealAcceptance.SnapshotRootCollision snapshotRoot) :
      Failure verify headers snapshotRoot

/-- Successful execution of one exact row-derived segment. -/
structure SegmentExecution
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (snapshotRoot : Snapshot -> Digest.Value)
    {before : ClosedCarry Digest.Value}
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) : Prop where
  initialRoot : snapshotRoot
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .initialSnapshot) = before.memoryRoot
  finalRoot : snapshotRoot
      (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
        .finalSnapshot) = run.after.memoryRoot
  executes : Memory.Executes
    (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
      .initialSnapshot).tuples
    before.globalTimestamp
    (ProductionMemoryRowSegments.accesses run.batches)
    (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
      .finalSnapshot).tuples
    run.after.globalTimestamp

namespace SegmentExecution

/-- Segment rows either give one exact port-aligned memory execution or name
the precise local obstruction. -/
theorem derive
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (snapshotRoot : Snapshot -> Digest.Value)
    {before : ClosedCarry Digest.Value}
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before) :
    Failure verify headers snapshotRoot \/
      SegmentExecution snapshotRoot run := by
  rcases ProductionMemorySegmentSoundness.SegmentRun.executesOrEvaluationFailure
      run with execution | fingerprintFailure
  · by_cases initialExact : snapshotRoot
        (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
          .initialSnapshot) = before.memoryRoot
    · by_cases finalExact : snapshotRoot
          (ProductionMemorySnapshotCoverage.SegmentRun.snapshot run
            .finalSnapshot) = run.after.memoryRoot
      · right
        refine ⟨initialExact, finalExact, ?_⟩
        have startExact := run.toProtocol.startsFromExactClosedCarry.2.1
        change run.active.globalTimestamp = before.globalTimestamp at startExact
        have accessesExact := run.consumed.accessesExact
        rw [startExact] at execution
        rw [← accessesExact]
        simpa [ProductionMemorySegmentSoundness.SegmentRun.accesses] using
          execution
      · exact Or.inl (.finalRoot run finalExact)
    · exact Or.inl (.initialRoot run initialExact)
  · exact Or.inl (.fingerprint run fingerprintFailure)

end SegmentExecution

/-- Successful nonempty global row-derived memory execution. -/
structure Execution
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (snapshotRoot : Snapshot -> Digest.Value)
    (initial : ClosedCarry Digest.Value)
    (batches : List (Evidence candidate schema verify headers))
    (final : ClosedCarry Digest.Value) where
  initialSnapshot : Snapshot
  finalSnapshot : Snapshot
  initialRoot : snapshotRoot initialSnapshot = initial.memoryRoot
  finalRoot : snapshotRoot finalSnapshot = final.memoryRoot
  executes : Memory.Executes initialSnapshot.tuples initial.globalTimestamp
    (ProductionMemoryRowSegments.accesses batches) finalSnapshot.tuples
    final.globalTimestamp

namespace Chain

/-- A nonempty exact segment chain is one application-port-aligned memory
execution unless a named local or boundary event occurs. -/
theorem executesOrFailure
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (snapshotRoot : Snapshot -> Digest.Value)
    {initial final : ClosedCarry Digest.Value}
    {batches : List (Evidence candidate schema verify headers)}
    {segmentCount : Nat}
    (chain : ProductionMemoryRowSegments.Chain candidate schema verify
      headers initial batches final segmentCount)
    (positive : 0 < segmentCount) :
    Failure verify headers snapshotRoot \/
      Nonempty (Execution snapshotRoot initial batches final) := by
  induction chain with
  | nil => omega
  | @cons before final tailBatches tailSegments head tail
      inductionHypothesis =>
      rcases SegmentExecution.derive snapshotRoot head with
        headFailure | headExecution
      · exact Or.inl headFailure
      · cases tail with
        | nil =>
            right
            refine ⟨
              { initialSnapshot :=
                  ProductionMemorySnapshotCoverage.SegmentRun.snapshot head
                    .initialSnapshot
                finalSnapshot :=
                  ProductionMemorySnapshotCoverage.SegmentRun.snapshot head
                    .finalSnapshot
                initialRoot := headExecution.initialRoot
                finalRoot := headExecution.finalRoot
                executes := ?_ }⟩
            simpa [ProductionMemoryRowSegments.accesses] using
              headExecution.executes
        | @cons tailBefore tailFinal restBatches restSegments tailHead
            tailRest =>
            have tailPositive : 0 < restSegments + 1 := by omega
            rcases inductionHypothesis tailPositive with
              tailFailure | tailExecutionNonempty
            · exact Or.inl tailFailure
            · rcases tailExecutionNonempty with ⟨tailExecution⟩
              let headFinal :=
                ProductionMemorySnapshotCoverage.SegmentRun.snapshot head
                  .finalSnapshot
              have equalRoot : snapshotRoot headFinal =
                  snapshotRoot tailExecution.initialSnapshot := by
                calc
                  snapshotRoot headFinal = head.after.memoryRoot :=
                    headExecution.finalRoot
                  _ = snapshotRoot tailExecution.initialSnapshot :=
                    tailExecution.initialRoot.symm
              by_cases boundaryExact :
                  headFinal = tailExecution.initialSnapshot
              · right
                refine ⟨
                  { initialSnapshot :=
                      ProductionMemorySnapshotCoverage.SegmentRun.snapshot head
                        .initialSnapshot
                    finalSnapshot := tailExecution.finalSnapshot
                    initialRoot := headExecution.initialRoot
                    finalRoot := tailExecution.finalRoot
                    executes := ?_ }⟩
                have tailExec := tailExecution.executes
                rw [← boundaryExact] at tailExec
                have joined := headExecution.executes.append
                  tailExec
                simpa only [accesses_append] using joined
              · exact Or.inl (.snapshotCollision
                  ⟨headFinal, tailExecution.initialSnapshot, boundaryExact,
                    equalRoot⟩)

end Chain

end Nightstream.Implementation.Nebula.ProductionMemoryChainSoundness
