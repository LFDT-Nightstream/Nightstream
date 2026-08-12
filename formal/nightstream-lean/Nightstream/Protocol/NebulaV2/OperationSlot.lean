import Nightstream.Protocol.NebulaV2.Fingerprint

/-!
Contract: independent semantics for one fixed-position V2 operation slot.

Assurance tier: protocol model.

Owns the exact operation payload meaning, canonical inactive padding, the
prefix-counter timestamp schedule, ROM/RAM rules, and conversion of an active
slot to one `Access.ValidAt` plus bounded read and write fingerprint tuples.

Does not own circuit rows, physical 3-by-21 routing, application instruction
semantics, or product accumulation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.OperationSlot

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Fingerprint

/-- Decoded values for one authority-bearing 106-bit operation slot. The
write timestamp and prefix counters are relation-derived values, not encoded
payload fields. -/
structure Value where
  pad : Nat
  isWrite : Nat
  isRam : Nat
  address : Nat
  readValue : Nat
  writeValue : Nat
  readTimestamp : Nat
  writeTimestamp : Nat
  countBefore : Nat
  countAfter : Nat
deriving DecidableEq, Repr

def Value.space (slot : Value) : MemorySpace :=
  if slot.isRam = 0 then .rom else .ram

def Value.kind (slot : Value) : AccessKind :=
  if slot.isWrite = 0 then .read else .write slot.writeValue

def Value.index (slot : Value) : Nat :=
  globalIndex slot.space slot.address

def Value.readTuple (slot : Value) : MemTuple :=
  { timestamp := slot.readTimestamp
    globalIndex := slot.index
    value := slot.readValue }

def Value.writeTuple (slot : Value) : MemTuple :=
  { timestamp := slot.writeTimestamp
    globalIndex := slot.index
    value := slot.writeValue }

def Value.access (slot : Value) : Access :=
  { space := slot.space
    address := slot.address
    kind := slot.kind
    read := slot.readTuple
    write := slot.writeTuple }

/-- Exact source conditions for one slot at a checked-step input timestamp.
No product value or endpoint occurs in this predicate. -/
structure ValidAt (slot : Value) (stepTimestampIn : Nat) : Prop where
  padBinary : slot.pad = 0 ∨ slot.pad = 1
  isWriteBinary : slot.isWrite = 0 ∨ slot.isWrite = 1
  isRamBinary : slot.isRam = 0 ∨ slot.isRam = 1
  addressBound : slot.address < 2 ^ 16
  readValueBound : slot.readValue < valueLimit
  writeValueBound : slot.writeValue < valueLimit
  readTimestampBound : slot.readTimestamp < timestampLimit
  countStep : slot.countAfter = slot.countBefore + (1 - slot.pad)
  countBeforeBound : slot.countBefore ≤ 63
  countAfterBound : slot.countAfter ≤ 63
  writeTimestampRule :
    slot.writeTimestamp = stepTimestampIn + slot.countAfter
  writeTimestampBound : slot.writeTimestamp < timestampLimit
  inactiveZero : slot.pad = 1 →
    slot.isWrite = 0 ∧ slot.isRam = 0 ∧ slot.address = 0 ∧
      slot.readValue = 0 ∧ slot.writeValue = 0 ∧
      slot.readTimestamp = 0
  readRule :
    slot.pad = 0 → slot.isWrite = 0 →
      slot.writeValue = slot.readValue
  romAddressBound :
    slot.pad = 0 → slot.isRam = 0 → slot.address < romCells
  noRomWrite :
    slot.pad = 0 → slot.isWrite = 1 → slot.isRam = 1
  readBeforeWrite :
    slot.pad = 0 → slot.readTimestamp < slot.writeTimestamp

theorem ValidAt.active_increment
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    slot.countAfter = slot.countBefore + 1 := by
  rw [valid.countStep, active]

theorem ValidAt.active_write_timestamp
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    slot.writeTimestamp = stepTimestampIn + slot.countBefore + 1 := by
  rw [valid.writeTimestampRule, valid.active_increment active]
  omega

theorem ValidAt.index_bound
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    slot.index < scannedCells := by
  rcases valid.isRamBinary with ram | ram
  · simp only [Value.index, Value.space, ram, ↓reduceIte, globalIndex]
    exact (valid.romAddressBound active ram).trans (by
      norm_num [scannedCells, romCells, ramCells])
  · simpa [Value.index, Value.space, ram, globalIndex, scannedCells,
      ramCells] using Nat.add_lt_add_left valid.addressBound romCells

theorem ValidAt.access_wellFormed
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    slot.access.WellFormed := by
  refine
    { addressInRange := ?_
      readIndex := rfl
      writeIndex := rfl
      readValueInRange := valid.readValueBound
      writeValueInRange := valid.writeValueBound
      valueRule := ?_ }
  · rcases valid.isRamBinary with ram | ram
    · simp only [Value.access, Value.space, ram, ↓reduceIte,
        MemorySpace.capacity]
      exact valid.romAddressBound active ram
    · simp only [Value.access, Value.space, ram, if_false,
        MemorySpace.capacity]
      simpa [ramCells] using valid.addressBound
  · rcases valid.isWriteBinary with write | write
    · simp only [Value.access, Value.kind, write, ↓reduceIte]
      exact valid.readRule active write
    · simp only [Value.access, Value.kind, write, if_false]
      refine ⟨?_, valid.writeValueBound, rfl⟩
      rcases valid.isRamBinary with ram | ram
      · exact False.elim (by
          have := valid.noRomWrite active write
          omega)
      · simp [Value.space, ram]

/-- An active slot is one exact scheduled semantic access. -/
theorem ValidAt.access_validAt
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    slot.access.ValidAt (stepTimestampIn + slot.countBefore) := by
  refine
    { wellFormed := valid.access_wellFormed active
      timestampInRange := ?_
      timestampOutRange := ?_
      readBeforeWrite := ?_
      writeTimestamp := ?_ }
  · have writeBound := valid.writeTimestampBound
    rw [valid.active_write_timestamp active] at writeBound
    omega
  · simpa [valid.active_write_timestamp active] using
      valid.writeTimestampBound
  · have strict := valid.readBeforeWrite active
    simpa [valid.active_write_timestamp active] using strict
  · simpa [Value.access, Value.writeTuple,
      valid.active_write_timestamp active]

theorem ValidAt.read_tuple_in_range
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    TupleInRange slot.readTuple := by
  exact ⟨valid.readTimestampBound, valid.index_bound active,
    valid.readValueBound⟩

theorem ValidAt.write_tuple_in_range
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    TupleInRange slot.writeTuple := by
  exact ⟨valid.writeTimestampBound, valid.index_bound active,
    valid.writeValueBound⟩

def ValidAt.readBounded
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    BoundedTuple :=
  ⟨slot.readTuple, valid.read_tuple_in_range active⟩

def ValidAt.writeBounded
    {slot : Value} {stepTimestampIn : Nat}
    (valid : ValidAt slot stepTimestampIn) (active : slot.pad = 0) :
    BoundedTuple :=
  ⟨slot.writeTuple, valid.write_tuple_in_range active⟩

end Nightstream.Protocol.NebulaV2.OperationSlot
