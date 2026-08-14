import Nightstream.Implementation.Nebula.Memory.Snapshot.SegmentCoverage

/-! Focused gates for row-derived full-snapshot coverage. -/

set_option autoImplicit false

namespace tests.NebulaSegmentSnapshotCoverage

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.Nebula.RecursiveManifestNifsCall
open Nightstream.Implementation.Nebula.RecursiveManifestSchema
open Nightstream.Implementation.Nebula.SegmentCheckedRows
open Nightstream.Implementation.Nebula.SegmentSnapshotCoverage
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.ProductState
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

end tests.NebulaSegmentSnapshotCoverage
