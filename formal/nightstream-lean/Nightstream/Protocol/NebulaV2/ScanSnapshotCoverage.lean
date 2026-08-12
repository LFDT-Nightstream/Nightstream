import Mathlib.Data.List.OfFn
import Nightstream.Protocol.NebulaV2.ScanSchedule

/-!
Contract: reconstruct one canonical snapshot from structural scan records.

Assurance tier: protocol model.

Owns the exact list equality between the 1,088-by-64 step-major scan and the
canonical 69,632-entry snapshot list. A record source can select only its
value and timestamp. Its global index must equal the verifier-fixed scan
position.

Does not own circuit rows, claim scheduling, boundary validity, commitments,
or fingerprint products.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.ScanSchedule

/-- Copy the value and timestamp from each scan record. The structural
position, not the source record, selects the snapshot index. -/
def cellsOfRecords (records : Position → MemTuple) : Position → CellState :=
  fun position =>
    { value := (records position).value
      lastTimestamp := (records position).timestamp }

/-- The unique canonical snapshot reconstructed through the scan bijection. -/
def snapshotOfRecords (records : Position → MemTuple) : Snapshot :=
  fun index => cellsOfRecords records (positionOfIndex index)

/-- The tuple selected by one structural scan position. -/
def tupleAtPosition (records : Position → MemTuple)
    (position : Position) : MemTuple :=
  { timestamp := (records position).timestamp
    globalIndex := position.globalIndex
    value := (records position).value }

theorem record_eq_tupleAtPosition
    {records : Position → MemTuple}
    (structural : ∀ position,
      (records position).globalIndex = position.globalIndex)
    (position : Position) :
    records position = tupleAtPosition records position := by
  apply MemTuple.ext
  · rfl
  · exact structural position
  · rfl

theorem tupleAtPosition_eq_snapshotTuple
    (records : Position → MemTuple) (position : Position) :
    tupleAtPosition records position =
      (snapshotOfRecords records).tupleAt position.globalIndex := by
  apply MemTuple.ext
  · simp [tupleAtPosition, Snapshot.tupleAt, snapshotOfRecords,
      cellsOfRecords]
  · rfl
  · simp [tupleAtPosition, Snapshot.tupleAt, snapshotOfRecords,
      cellsOfRecords]

/-- A step-major list of structural scan tuples is exactly the canonical
snapshot tuple list. This is an equality of ordered lists, not only a set or
cardinality statement. -/
theorem nestedTupleList_eq_snapshotTupleList
    (records : Position → MemTuple) :
    (List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots =>
        tupleAtPosition records ⟨step, slot⟩).flatten =
      (snapshotOfRecords records).tupleList := by
  let flat : Fin (claimsPerSegment * scanSlots) → MemTuple :=
    fun index =>
      (snapshotOfRecords records).tupleAt
        (Fin.cast scan_capacity index)
  have split := List.ofFn_mul flat
  calc
    (List.ofFn fun step : Fin claimsPerSegment =>
        List.ofFn fun slot : Fin scanSlots =>
          tupleAtPosition records ⟨step, slot⟩).flatten =
        (List.ofFn fun step : Fin claimsPerSegment =>
          List.ofFn fun slot : Fin scanSlots =>
            flat ⟨step.val * scanSlots + slot.val, by
              calc
                step.val * scanSlots + slot.val <
                    (step.val + 1) * scanSlots :=
                  (Nat.add_lt_add_left slot.isLt _).trans_eq (by
                    rw [Nat.add_mul, Nat.one_mul])
                _ ≤ claimsPerSegment * scanSlots :=
                  Nat.mul_le_mul_right scanSlots step.isLt⟩).flatten := by
      apply congrArg List.flatten
      apply List.ofFn_inj.mpr
      funext step
      apply List.ofFn_inj.mpr
      funext slot
      have exactTuple := tupleAtPosition_eq_snapshotTuple records
        (Position.mk step slot)
      rw [exactTuple]
      apply congrArg ((snapshotOfRecords records).tupleAt)
        (Fin.ext rfl)
    _ = List.ofFn flat := split.symm
    _ = (snapshotOfRecords records).tupleList := by
      simp only [Snapshot.tupleList]
      rw [List.ofFn_congr scan_capacity flat]
      congr 1

/-- If each source record uses its structural address, all source records are
exactly the reconstructed canonical snapshot, including multiplicity. -/
theorem nestedRecords_eq_snapshotTupleList
    {records : Position → MemTuple}
    (structural : ∀ position,
      (records position).globalIndex = position.globalIndex) :
    (List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots => records ⟨step, slot⟩).flatten =
      (snapshotOfRecords records).tupleList := by
  rw [show
    (List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots => records ⟨step, slot⟩).flatten =
    (List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots =>
        tupleAtPosition records ⟨step, slot⟩).flatten by
      congr 2
      funext step
      congr 1
      funext slot
      exact record_eq_tupleAtPosition structural ⟨step, slot⟩]
  exact nestedTupleList_eq_snapshotTupleList records

theorem nestedRecords_eq_snapshotTuples
    {records : Position → MemTuple}
    (structural : ∀ position,
      (records position).globalIndex = position.globalIndex) :
    ((List.ofFn fun step : Fin claimsPerSegment =>
      List.ofFn fun slot : Fin scanSlots => records ⟨step, slot⟩).flatten :
        Multiset MemTuple) =
      (snapshotOfRecords records).tuples := by
  exact congrArg (fun values : List MemTuple => (values : Multiset MemTuple))
    (nestedRecords_eq_snapshotTupleList structural)

end Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage
