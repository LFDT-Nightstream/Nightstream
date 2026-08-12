import Nightstream.Implementation.NebulaV2.Memory.Segment.Coverage

/-! Focused gates for complete row-derived segment memory coverage. -/

set_option autoImplicit false

namespace tests.NebulaV2SegmentMemoryCoverage

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

theorem all_invocation_operations_are_covered
    (invocations : List (Invocation artifact selected)) :
    ((Run.chunks invocations).map ProductState.Chunk.writes).sum =
        (Memory.writeTuples (accesses invocations) : Multiset MemTuple) ∧
      ((Run.chunks invocations).map ProductState.Chunk.reads).sum =
        (Memory.readTuples (accesses invocations) : Multiset MemTuple) :=
  ⟨writesCover invocations, readsCover invocations⟩

theorem one_closed_segment_has_complete_coverage
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
      (Run.chunks invocations) :=
  covers run startsAtZero

theorem one_closed_segment_is_strictly_ordered
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed)) :
    Ordered active.globalTimestamp (accesses invocations)
      closed.globalTimestamp :=
  orderedActiveToClosed run

end tests.NebulaV2SegmentMemoryCoverage
