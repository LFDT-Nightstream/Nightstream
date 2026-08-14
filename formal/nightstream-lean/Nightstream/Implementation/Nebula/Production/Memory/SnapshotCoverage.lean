import Nightstream.Implementation.Nebula.Production.Memory.RowSegments
import Nightstream.Protocol.Nebula.ScanSnapshotCoverage

/-!
Contract: reconstruct both complete canonical snapshots from one row-derived
production memory segment.

The segment theorem fixes all 1,088 checked-step positions. Each checked step
fixes all 64 snapshot slots. This file composes those facts through the scan
bijection and derives one 69,632-cell initial snapshot and one 69,632-cell
final snapshot, including exact list multiplicity and boundary validity.

No snapshot, address list, or coverage predicate is supplied by the caller.

Does not own fingerprint accumulation, root binding, application alignment,
or deployed-verifier extraction.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage

open Nightstream.Implementation.Nebula.ProductionMemoryRowSegments
open Nightstream.Implementation.Nebula.ProductionMemoryStepSemantics
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.ScanSchedule
open Nightstream.Protocol.Nebula.ScanSnapshotCoverage
open Nightstream.SuperNeo.Concrete

def chunkSnapshot (chunk : ProductState.Chunk) :
    SnapshotRole -> Multiset MemTuple
  | .initialSnapshot => chunk.initialSnapshot
  | .finalSnapshot => chunk.finalSnapshot

/-- The chunk multisets of a checked-step list equal its flattened complete
snapshot lists. -/
theorem snapshotChunkSum_eq_flattenedLists
    (checked : List ProductionMemoryStepSemantics.Step)
    (role : SnapshotRole) :
    ((ProductionMemoryStepSemantics.Run.chunks checked).map fun chunk =>
      chunkSnapshot chunk role).sum =
      ((ProductionMemoryStepSemantics.Run.snapshotLists checked role).flatten :
        Multiset MemTuple) := by
  induction checked with
  | nil => simp [ProductionMemoryStepSemantics.Run.chunks,
      ProductionMemoryStepSemantics.Run.snapshotLists]
  | cons head tail inductionHypothesis =>
      cases role <;>
        simp only [ProductionMemoryStepSemantics.Run.chunks, List.map_cons,
          List.sum_cons, ProductionMemoryStepSemantics.Run.snapshotLists,
          List.flatten_cons, chunkSnapshot] at inductionHypothesis ⊢
      · have headExact :
            (head.snapshotList .initialSnapshot : Multiset MemTuple) =
              head.records.chunk.initialSnapshot := by
          simpa using head.snapshotList_coe_eq_chunk .initialSnapshot
        rw [← headExact]
        rw [inductionHypothesis]
        simp
      · have headExact :
            (head.snapshotList .finalSnapshot : Multiset MemTuple) =
              head.records.chunk.finalSnapshot := by
          simpa using head.snapshotList_coe_eq_chunk .finalSnapshot
        rw [← headExact]
        rw [inductionHypothesis]
        simp

namespace SegmentRun

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}
variable {before : ClosedCarry Digest.Value}

/-- Convert a fixed scan-step position to the corresponding row-derived
checked-step list position. -/
def indexAt
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (step : Fin claimsPerSegment) : Fin (steps run.batches).length :=
  ⟨step.val, by
    rw [run.exactStepCount]
    exact step.isLt⟩

def stepAt
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (step : Fin claimsPerSegment) : ProductionMemoryStepSemantics.Step :=
  (steps run.batches).get (indexAt run step)

theorem stepAt_stepIndex
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (step : Fin claimsPerSegment) :
    (stepAt run step).claim.stepIndex.val = step.val := by
  simpa [stepAt, indexAt] using run.stepIndexAt (indexAt run step)

theorem stepAt_segmentBounds
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (step : Fin claimsPerSegment) :
    (stepAt run step).claim.segmentStartTimestamp =
        run.active.segmentStartTimestamp /\
      (stepAt run step).claim.segmentEndTimestamp =
        run.active.segmentEndTimestamp := by
  simpa [stepAt, indexAt] using run.segmentBoundsAt (indexAt run step)

def activeBoundary
    (active : ActiveCarry Digest.Value (ProductState.Challenges K)
      (ProductState.State K)) : SnapshotRole -> Nat
  | .initialSnapshot => active.segmentStartTimestamp
  | .finalSnapshot => active.segmentEndTimestamp

theorem stepAt_boundaryValue
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (step : Fin claimsPerSegment) (role : SnapshotRole) :
    SnapshotSlotRows.boundaryValue (stepAt run step).claim role =
      activeBoundary run.active role := by
  have bounds := stepAt_segmentBounds run step
  cases role
  · simpa [SnapshotSlotRows.boundaryValue, activeBoundary] using bounds.1
  · simpa [SnapshotSlotRows.boundaryValue, activeBoundary] using bounds.2

/-- Row-derived snapshot records indexed by the verifier-fixed scan
position. -/
def snapshotRecords
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) : Position -> MemTuple :=
  fun position =>
    ((stepAt run position.step).records.snapshot role position.slot).1

theorem snapshotRecords_structural
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) (position : Position) :
    (snapshotRecords run role position).globalIndex = position.globalIndex := by
  simp only [snapshotRecords]
  rw [(stepAt run position.step).snapshotGlobalIndex role position.slot]
  rw [stepAt_stepIndex run position.step]
  rfl

/-- The unique complete canonical snapshot reconstructed from all row
records for one role. -/
def snapshot
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) : Snapshot :=
  ScanSnapshotCoverage.snapshotOfRecords (snapshotRecords run role)

/-- Every reconstructed cell has the row-proved value bound and the exact
segment-boundary timestamp bound. -/
theorem snapshotValidAt
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) :
    Snapshot.ValidAt (snapshot run role) (activeBoundary run.active role) := by
  intro index
  change
    (snapshotRecords run role (positionOfIndex index)).value < valueLimit /\
      (snapshotRecords run role (positionOfIndex index)).timestamp <=
        activeBoundary run.active role
  have valid := (stepAt run (positionOfIndex index).step).snapshotValid role
    (positionOfIndex index).slot
  rw [stepAt_boundaryValue run (positionOfIndex index).step role] at valid
  exact valid

private theorem mappedSnapshotLists_eq_nested
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) :
    ProductionMemoryStepSemantics.Run.snapshotLists (steps run.batches) role =
      List.ofFn (fun step : Fin claimsPerSegment =>
        List.ofFn fun slot : Fin scanSlots =>
          snapshotRecords run role ⟨step, slot⟩) := by
  have exactCount := run.exactStepCount
  calc
    ProductionMemoryStepSemantics.Run.snapshotLists (steps run.batches) role =
        List.ofFn (fun index : Fin (steps run.batches).length =>
          ((steps run.batches).get index).snapshotList role) := by
      exact (List.ofFn_getElem_eq_map (steps run.batches)
        (fun checked => checked.snapshotList role)).symm
    _ = List.ofFn (fun step : Fin claimsPerSegment =>
        List.ofFn fun slot : Fin scanSlots =>
          snapshotRecords run role ⟨step, slot⟩) := by
      rw [List.ofFn_congr exactCount]
      apply List.ofFn_inj.mpr
      funext step
      apply List.ofFn_inj.mpr
      funext slot
      rfl

/-- The row-derived chunk sum is exactly one complete canonical snapshot
multiset. Omission, duplication, and address selection are conclusions, not
premises. -/
theorem snapshotChunksCover
    (run : ProductionMemoryRowSegments.SegmentRun candidate schema verify
      headers before)
    (role : SnapshotRole) :
    ((ProductionMemoryStepSemantics.Run.chunks (steps run.batches)).map
      fun chunk => chunkSnapshot chunk role).sum =
        (snapshot run role).tuples := by
  rw [snapshotChunkSum_eq_flattenedLists]
  have nested := congrArg List.flatten
    (mappedSnapshotLists_eq_nested run role)
  rw [nested]
  exact nestedRecords_eq_snapshotTuples
    (snapshotRecords_structural run role)

end SegmentRun

end Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage
