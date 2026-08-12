import Nightstream.Implementation.NebulaV2.Memory.Snapshot.SegmentCoverage

/-!
Contract: complete row-derived memory coverage for one V2 segment.

Assurance tier: implementation-to-protocol bridge.

Owns the exact ordered application-access list across all checked invocations,
its strict global timestamp schedule, and the four exact chunk-coverage
equalities for the reconstructed initial and final snapshots.

Does not own generated WASM transition-row soundness, root-chain binding,
fingerprint probability, NIFS row soundness, or terminal verification.

Emits constraints: no. It gives aggregate meaning to existing checked rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

/-- The only segment access list is the physical-order concatenation of the
row-derived access list from every checked invocation. -/
def accesses (invocations : List (Invocation artifact selected)) :
    List Access :=
  invocations.flatMap Invocation.applicationAccesses

/-- All write-source chunks equal the writes of the exact application access
list, including multiset multiplicity. -/
theorem writesCover
    (invocations : List (Invocation artifact selected)) :
    ((Run.chunks invocations).map ProductState.Chunk.writes).sum =
      (Memory.writeTuples (accesses invocations) : Multiset MemTuple) := by
  induction invocations with
  | nil => simp [Run.chunks, accesses, Memory.writeTuples]
  | cons head tail inductionHypothesis =>
      simp only [Run.chunks, List.map_cons, List.sum_cons, accesses,
        List.flatMap_cons]
      rw [head.chunk_writes_eq]
      have tailExact :
          (List.map ProductState.Chunk.writes
            (List.map Invocation.chunk tail)).sum =
            (Memory.writeTuples
              (List.flatMap Invocation.applicationAccesses tail) :
                Multiset MemTuple) := by
        simpa [Run.chunks, accesses] using inductionHypothesis
      rw [tailExact]
      simp [Memory.writeTuples]

/-- All read-source chunks equal the reads of the exact application access
list. -/
theorem readsCover
    (invocations : List (Invocation artifact selected)) :
    ((Run.chunks invocations).map ProductState.Chunk.reads).sum =
      (Memory.readTuples (accesses invocations) : Multiset MemTuple) := by
  induction invocations with
  | nil => simp [Run.chunks, accesses, Memory.readTuples]
  | cons head tail inductionHypothesis =>
      simp only [Run.chunks, List.map_cons, List.sum_cons, accesses,
        List.flatMap_cons]
      rw [head.chunk_reads_eq]
      have tailExact :
          (List.map ProductState.Chunk.reads
            (List.map Invocation.chunk tail)).sum =
            (Memory.readTuples
              (List.flatMap Invocation.applicationAccesses tail) :
                Multiset MemTuple) := by
        simpa [Run.chunks, accesses] using inductionHypothesis
      rw [tailExact]
      simp [Memory.readTuples]

/-- Exact carry chaining joins every invocation-local timestamp schedule into
one strict segment schedule. -/
theorem ordered
    {before after : ConcreteCarry}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected before invocations after) :
    Ordered (carryTimestamp before) (accesses invocations)
      (carryTimestamp after) := by
  induction run with
  | nil => exact .nil _
  | cons head rest inductionHypothesis =>
      have headOrdered := head.applicationAccessesOrdered
      have startExact := head.transition.consumes.timestampIn_eq_before
      have endExact := head.transition.consumes.timestampOut_eq_after
      change head.call.claim.memory.timestampIn =
        carryTimestamp head.beforeCarry at startExact
      change head.call.claim.memory.timestampOut =
        carryTimestamp head.afterCarry at endExact
      rw [startExact, endExact] at headOrdered
      simpa [accesses] using headOrdered.append inductionHypothesis

/-- The complete row-derived segment covers both canonical snapshots and all
application operations exactly once. -/
theorem covers
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0) :
    ProductState.Covers
      (SegmentSnapshotCoverage.CheckedRun.snapshot run startsAtZero
        .initialSnapshot)
      (accesses invocations)
      (SegmentSnapshotCoverage.CheckedRun.snapshot run startsAtZero
        .finalSnapshot)
      (Run.chunks invocations) where
  initialSnapshot := by
    simpa [SegmentSnapshotCoverage.chunkSnapshot] using
      SegmentSnapshotCoverage.CheckedRun.snapshotChunksCover
        run startsAtZero .initialSnapshot
  writes := writesCover invocations
  reads := readsCover invocations
  finalSnapshot := by
    simpa [SegmentSnapshotCoverage.chunkSnapshot] using
      SegmentSnapshotCoverage.CheckedRun.snapshotChunksCover
        run startsAtZero .finalSnapshot

/-- The opening-to-close timestamp schedule is the exact ordered schedule of
the row-derived application accesses. -/
theorem orderedActiveToClosed
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed)) :
    Ordered active.globalTimestamp (accesses invocations)
      closed.globalTimestamp := by
  simpa [carryTimestamp] using ordered run

end Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage
