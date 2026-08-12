import Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage

/-! Focused gates for row-derived full-snapshot coverage. -/

set_option autoImplicit false

namespace tests.NebulaV2SegmentSnapshotCoverage

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.NebulaV2.SegmentCheckedRows
open Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

theorem checked_segment_covers_one_snapshot
    {widths : CompilerWidths}
    {artifact : Artifact widths} {selected : SelectedVerifier widths}
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) :
    ((Run.chunks invocations).map fun chunk => chunkSnapshot chunk role).sum =
      (CheckedRun.snapshot run startsAtZero role).tuples :=
  CheckedRun.snapshotChunksCover run startsAtZero role

theorem checked_segment_snapshot_is_boundary_valid
    {widths : CompilerWidths}
    {artifact : Artifact widths} {selected : SelectedVerifier widths}
    {active : ActiveCarry Digest.Value
      (ProductState.Challenges K) (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0)
    (role : SnapshotRole) :
    Snapshot.ValidAt (CheckedRun.snapshot run startsAtZero role)
      (CheckedRun.activeBoundary active role) :=
  CheckedRun.snapshotValidAt run startsAtZero role

end tests.NebulaV2SegmentSnapshotCoverage
