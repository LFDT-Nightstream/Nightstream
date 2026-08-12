import Nightstream.Implementation.NebulaV2.Memory.Segment.CheckedRows
import Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage

/-!
Contract: derive complete canonical snapshots from one checked segment.

Assurance tier: implementation-to-protocol bridge.

Owns the connection from all row-derived initial and final snapshot slots in
one exact 1,088-invocation run to two function-valued canonical snapshots and
their exact segment-boundary validity. The proof derives step order and both
boundaries from delayed full-claim consumption and derives each address from
the fixed 64-slot row layout.

Does not own application-port coverage, root-chain binding, NIFS row
soundness, fingerprint probability, or the terminal verifier.

Emits constraints: no. It gives aggregate meaning to existing rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge
open Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ScanSchedule
open Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage
open Nightstream.SuperNeo.Concrete

def chunkSnapshot (chunk : ProductState.Chunk) :
    SnapshotRole → Multiset MemTuple
  | .initialSnapshot => chunk.initialSnapshot
  | .finalSnapshot => chunk.finalSnapshot

def snapshotList
    {widths : CompilerWidths}
    {artifact : Artifact widths} {selected : SelectedVerifier widths}
    (invocation : Invocation artifact selected) (role : SnapshotRole) :
    List MemTuple :=
  List.ofFn fun slot : Fin ScanSchedule.scanSlots =>
    (invocation.source.records.snapshot role slot).1

private theorem activeRecords_snapshotRecords
    (records : Fin ScanSchedule.scanSlots → BoundedTuple) :
    activeRecords (snapshotRecords records) = List.ofFn records := by
  rw [show snapshotRecords records =
      (List.ofFn records).map some by
    rw [List.map_ofFn]
    rfl]
  simp [activeRecords]

theorem snapshotList_coe_eq_chunkSnapshot
    {widths : CompilerWidths}
    {artifact : Artifact widths} {selected : SelectedVerifier widths}
    (invocation : Invocation artifact selected) (role : SnapshotRole) :
    (SegmentSnapshotCoverage.snapshotList invocation role :
      Multiset MemTuple) =
      chunkSnapshot invocation.chunk role := by
  cases role <;>
    simp only [SegmentSnapshotCoverage.snapshotList, chunkSnapshot,
      Invocation.chunk,
      CheckedStepRecords.chunk]
  all_goals
    simp only [activeRecordMultiset]
    rw [SegmentSnapshotCoverage.activeRecords_snapshotRecords]
    rfl

theorem snapshotChunkSum_eq_flattenedLists
    {widths : CompilerWidths}
    {artifact : Artifact widths} {selected : SelectedVerifier widths}
    (invocations : List (Invocation artifact selected))
    (role : SnapshotRole) :
    ((Run.chunks invocations).map fun chunk => chunkSnapshot chunk role).sum =
      ((invocations.map fun invocation =>
        SegmentSnapshotCoverage.snapshotList invocation role).flatten :
          Multiset MemTuple) := by
  induction invocations with
  | nil => simp [Run.chunks]
  | cons head tail inductionHypothesis =>
      simp only [Run.chunks, List.map_cons, List.sum_cons,
        List.flatten_cons]
      rw [← snapshotList_coe_eq_chunkSnapshot head role]
      have tailExact := inductionHypothesis
      simp only [Run.chunks] at tailExact
      rw [tailExact]
      simp

namespace CheckedRun

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

def indexAt
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (step : Fin claimsPerSegment) : Fin invocations.length :=
  ⟨step.val, by
    rw [run.exactClaimCount startsAtZero]
    exact step.isLt⟩

def invocationAt
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (step : Fin claimsPerSegment) : Invocation artifact selected :=
  invocations.get (CheckedRun.indexAt run startsAtZero step)

theorem invocationAt_stepIndex
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (step : Fin claimsPerSegment) :
    (CheckedRun.invocationAt run startsAtZero step).call.claim.memory.stepIndex.val =
      step.val := by
  have indexed := verifiedRun_claim_step_at run.toVerifiedRun active rfl
    (Fin.cast (by
      simp [SegmentCheckedRows.Run.verifiedClaims])
      (CheckedRun.indexAt run startsAtZero step))
  rw [startsAtZero] at indexed
  simpa [SegmentCheckedRows.Run.verifiedClaims, invocationAt, indexAt,
    Invocation.verified] using
    indexed

theorem invocationAt_segmentBounds
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (step : Fin claimsPerSegment) :
    (CheckedRun.invocationAt run startsAtZero step).call.claim.memory.segmentStartTimestamp =
        active.segmentStartTimestamp ∧
      (CheckedRun.invocationAt run startsAtZero step).call.claim.memory.segmentEndTimestamp =
        active.segmentEndTimestamp := by
  have indexed := verifiedRun_claim_segment_bounds_at run.toVerifiedRun
    active rfl
    (Fin.cast (by
      simp [SegmentCheckedRows.Run.verifiedClaims])
      (CheckedRun.indexAt run startsAtZero step))
  simpa [SegmentCheckedRows.Run.verifiedClaims, invocationAt, indexAt,
    Invocation.verified] using indexed

def activeBoundary
    (active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)) :
    SnapshotRole → Nat
  | .initialSnapshot => active.segmentStartTimestamp
  | .finalSnapshot => active.segmentEndTimestamp

theorem invocationAt_boundaryValue
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (step : Fin claimsPerSegment) (role : SnapshotRole) :
    SnapshotSlotRows.boundaryValue
        (CheckedRun.invocationAt run startsAtZero step).call.claim.memory role =
      activeBoundary active role := by
  have bounds := CheckedRun.invocationAt_segmentBounds run startsAtZero step
  cases role
  · simpa [SnapshotSlotRows.boundaryValue, activeBoundary] using bounds.1
  · simpa [SnapshotSlotRows.boundaryValue, activeBoundary] using bounds.2

def snapshotRecords
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) : Position → MemTuple :=
  fun position =>
    ((CheckedRun.invocationAt run startsAtZero position.step).source.records.snapshot
      role position.slot).1

private theorem invocationSnapshotRecord_globalIndex
    (invocation : Invocation artifact selected)
    (role : SnapshotRole) (slot : Fin ScanSchedule.scanSlots) :
    (invocation.source.records.snapshot role slot).1.globalIndex =
      SnapshotSlot.globalIndex invocation.call.claim.memory.stepIndex.val
        slot := by
  cases role <;> rfl

theorem snapshotRecords_structural
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) (position : Position) :
    (CheckedRun.snapshotRecords run startsAtZero role position).globalIndex =
      position.globalIndex := by
  simp only [CheckedRun.snapshotRecords]
  rw [invocationSnapshotRecord_globalIndex]
  rw [CheckedRun.invocationAt_stepIndex run startsAtZero position.step]
  rfl

private theorem invocationSnapshotRecord_valid
    (invocation : Invocation artifact selected)
    (role : SnapshotRole) (slot : Fin ScanSchedule.scanSlots) :
    let record := (invocation.source.records.snapshot role slot).1
    record.value < valueLimit ∧
      record.timestamp ≤
        SnapshotSlotRows.boundaryValue invocation.call.claim.memory role := by
  have valid := invocation.source.snapshot.valid role slot
  have cellValid := valid.cell_valid
  cases role <;>
    simpa [MemorySourceRows.Sound.records,
      CheckedStepRecords.snapshot, SnapshotChunkRows.Sound.records,
      SnapshotSlot.ValidAt.boundedTuple, SnapshotSlot.Value.tuple,
      SnapshotSlot.ValidAt.cellState] using cellValid

def snapshot
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) : Snapshot :=
  ScanSnapshotCoverage.snapshotOfRecords
    (CheckedRun.snapshotRecords run startsAtZero role)

/-- The same slot rows that fix full coverage also derive each cell's value
bound and its timestamp bound against the opening carry's exact boundary. -/
theorem snapshotValidAt
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) :
    Snapshot.ValidAt (CheckedRun.snapshot run startsAtZero role)
      (activeBoundary active role) := by
  intro index
  change
    (CheckedRun.snapshotRecords run startsAtZero role
        (positionOfIndex index)).value < valueLimit ∧
      (CheckedRun.snapshotRecords run startsAtZero role
        (positionOfIndex index)).timestamp ≤ activeBoundary active role
  have valid := invocationSnapshotRecord_valid
    (CheckedRun.invocationAt run startsAtZero
      (positionOfIndex index).step) role (positionOfIndex index).slot
  rw [CheckedRun.invocationAt_boundaryValue run startsAtZero
    (positionOfIndex index).step role] at valid
  exact valid

private theorem mappedSnapshotLists_eq_nested
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) :
    invocations.map (fun invocation =>
      SegmentSnapshotCoverage.snapshotList invocation role) =
      List.ofFn (fun step : Fin claimsPerSegment =>
        List.ofFn fun slot : Fin ScanSchedule.scanSlots =>
          CheckedRun.snapshotRecords run startsAtZero role ⟨step, slot⟩) := by
  have exactCount := run.exactClaimCount startsAtZero
  calc
    invocations.map (fun invocation =>
      SegmentSnapshotCoverage.snapshotList invocation role) =
        List.ofFn (fun index : Fin invocations.length =>
          SegmentSnapshotCoverage.snapshotList (invocations.get index)
            role) := by
      exact (List.ofFn_getElem_eq_map invocations
        (fun invocation =>
          SegmentSnapshotCoverage.snapshotList invocation role)).symm
    _ = List.ofFn (fun step : Fin claimsPerSegment =>
        List.ofFn fun slot : Fin ScanSchedule.scanSlots =>
          CheckedRun.snapshotRecords run startsAtZero role ⟨step, slot⟩) := by
      rw [List.ofFn_congr exactCount]
      apply List.ofFn_inj.mpr
      funext step
      apply List.ofFn_inj.mpr
      funext slot
      rfl

/-- All row-derived records for one snapshot role equal one complete
canonical snapshot multiset. Omission, duplication, and address selection are
not premises. -/
theorem snapshotChunksCover
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) :
    ((Run.chunks invocations).map fun chunk => chunkSnapshot chunk role).sum =
      (CheckedRun.snapshot run startsAtZero role).tuples := by
  rw [snapshotChunkSum_eq_flattenedLists]
  have nested := congrArg List.flatten
    (mappedSnapshotLists_eq_nested run startsAtZero role)
  rw [nested]
  exact nestedRecords_eq_snapshotTuples
    (snapshotRecords_structural run startsAtZero role)

end CheckedRun

end Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage
