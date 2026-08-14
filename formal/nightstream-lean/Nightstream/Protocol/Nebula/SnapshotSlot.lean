import Nightstream.Protocol.Nebula.Lifecycle
import Nightstream.Protocol.Nebula.Snapshot

/-!
Contract: independent meaning of one structural snapshot slot in one V2
checked step.

Assurance tier: protocol model.

Owns the value and timestamp payload, the structural
`step_index * 64 + slot` address, the segment-boundary timestamp condition,
and conversion to one bounded fingerprint tuple.

Does not own circuit rows, product accumulation, full-scan scheduling, or
boundary-snapshot continuity.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.SnapshotSlot

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

structure Value where
  value : Nat
  timestamp : Nat
deriving DecidableEq, Repr

def globalIndex (stepIndex : Nat) (slot : Fin 64) : Nat :=
  stepIndex * 64 + slot.val

def Value.tuple (source : Value) (stepIndex : Nat) (slot : Fin 64) :
    MemTuple :=
  { timestamp := source.timestamp
    globalIndex := globalIndex stepIndex slot
    value := source.value }

/-- Exact challenge-independent source conditions for one snapshot cell. -/
structure ValidAt
    (source : Value) (stepIndex boundaryTimestamp : Nat) : Prop where
  stepIndexBound : stepIndex < Lifecycle.claimsPerSegment
  valueBound : source.value < valueLimit
  timestampBound : source.timestamp < timestampLimit
  boundaryBound : boundaryTimestamp < timestampLimit
  timestampLeBoundary : source.timestamp ≤ boundaryTimestamp

theorem ValidAt.global_index_bound
    {source : Value} {stepIndex boundaryTimestamp : Nat}
    (valid : ValidAt source stepIndex boundaryTimestamp)
    (slot : Fin 64) :
    globalIndex stepIndex slot < scannedCells := by
  have slotBound := slot.isLt
  have stepBound := valid.stepIndexBound
  norm_num [Lifecycle.claimsPerSegment, globalIndex, scannedCells,
    romCells, ramCells] at stepBound slotBound ⊢
  omega

theorem ValidAt.tuple_in_range
    {source : Value} {stepIndex boundaryTimestamp : Nat}
    (valid : ValidAt source stepIndex boundaryTimestamp)
    (slot : Fin 64) :
    TupleInRange (source.tuple stepIndex slot) := by
  exact ⟨valid.timestampBound, valid.global_index_bound slot,
    valid.valueBound⟩

def ValidAt.boundedTuple
    {source : Value} {stepIndex boundaryTimestamp : Nat}
    (valid : ValidAt source stepIndex boundaryTimestamp)
    (slot : Fin 64) : BoundedTuple :=
  ⟨source.tuple stepIndex slot, valid.tuple_in_range slot⟩

def ValidAt.cellState
    {source : Value} {stepIndex boundaryTimestamp : Nat}
    (_valid : ValidAt source stepIndex boundaryTimestamp) : CellState :=
  { value := source.value
    lastTimestamp := source.timestamp }

theorem ValidAt.cell_valid
    {source : Value} {stepIndex boundaryTimestamp : Nat}
    (valid : ValidAt source stepIndex boundaryTimestamp) :
    (valid.cellState.value < valueLimit) ∧
      valid.cellState.lastTimestamp ≤ boundaryTimestamp := by
  exact ⟨valid.valueBound, valid.timestampLeBoundary⟩

end Nightstream.Protocol.Nebula.SnapshotSlot
