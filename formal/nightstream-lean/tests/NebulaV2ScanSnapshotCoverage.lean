import Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage

/-! Focused gates for canonical full-snapshot reconstruction. -/

set_option autoImplicit false

namespace tests.NebulaV2ScanSnapshotCoverage

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ScanSchedule
open Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage

theorem structural_records_are_one_canonical_snapshot
    {records : Position → MemTuple}
    (structural : ∀ position,
      (records position).globalIndex = position.globalIndex) :
    ((List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots => records ⟨step, slot⟩).flatten :
        Multiset MemTuple) =
      (snapshotOfRecords records).tuples :=
  nestedRecords_eq_snapshotTuples structural

theorem structural_records_preserve_order
    {records : Position → MemTuple}
    (structural : ∀ position,
      (records position).globalIndex = position.globalIndex) :
    (List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots => records ⟨step, slot⟩).flatten =
      (snapshotOfRecords records).tupleList :=
  nestedRecords_eq_snapshotTupleList structural

end tests.NebulaV2ScanSnapshotCoverage
