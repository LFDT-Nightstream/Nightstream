import Nightstream.Protocol.Nebula.ScanSnapshotCoverage

/-! Focused gates for canonical full-snapshot reconstruction. -/

set_option autoImplicit false

namespace tests.NebulaScanSnapshotCoverage

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.ScanSchedule
open Nightstream.Protocol.Nebula.ScanSnapshotCoverage

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

end tests.NebulaScanSnapshotCoverage
