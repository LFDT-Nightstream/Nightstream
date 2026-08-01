import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory
import Nightstream.Implementation.Lowering.Nebula.BitSemantics

/-!
Contract: bind the two benchmark batches to the Lean-owned Nebula columns.

Assurance tier: model-level.

This file names the physical public and witness words used by each batch and
derives the exact protocol tuples decoded from them. It does not accept tuple
lists or product conclusions as premises.

It does not own an honest physical assignment, product chaining between the
two batches, the combined CCS relation, F-prime assembly, Rust, or a security
reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaBinding

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceProducts
open Nightstream.Implementation.Lowering.Nebula.BitSemantics
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

private abbrev Lin := Rows.LinearCombination

def Carries
    (assignment : Nat -> F) (combination : Lin) (value : Nat) : Prop :=
  fieldValue assignment combination = Compiler.fieldOfNat value

def initialRomAt (slot : Nat) : MemTuple :=
  Memory.blankCell slot

def finalRomAt (slot : Nat) : MemTuple :=
  Memory.blankCell slot

/-- Source words for the first batch: one active RAM read at address zero and
the first 1,024-cell scan chunk. -/
structure FirstPorts (assignment : Nat -> F) : Prop where
  constantWire : assignment 0 = 1
  step : Carries assignment
    (publicWord XOffset.step stepIndexBits) 0
  timestampIn : Carries assignment Compiler.timestampIn 0
  timestampOut : Carries assignment
    (publicWord XOffset.timestampOut Layout.timestampBits) 1
  pad : Carries assignment (operationPad wasm42x6 0) 0
  isWrite : Carries assignment (operationIsWrite wasm42x6 0) 0
  ram : Carries assignment (operationRam wasm42x6 0) 1
  address : Carries assignment (operationAddress wasm42x6 0) 0
  readValue : Carries assignment (operationReadValue wasm42x6 0) 42
  writeValue : Carries assignment (operationWriteValue wasm42x6 0) 42
  readTimestamp : Carries assignment
    (operationReadTimestamp wasm42x6 0) 0
  initialScanValue : forall slot, slot < 1024 ->
    Carries assignment (scanValue wasm42x6 false slot)
      (initialRomAt slot).value
  initialScanTimestamp : forall slot, slot < 1024 ->
    Carries assignment (scanTimestamp wasm42x6 false slot)
      (initialRomAt slot).timestamp
  finalScanValue : forall slot, slot < 1024 ->
    Carries assignment (scanValue wasm42x6 true slot)
      (finalRomAt slot).value
  finalScanTimestamp : forall slot, slot < 1024 ->
    Carries assignment (scanTimestamp wasm42x6 true slot)
      (finalRomAt slot).timestamp

/-- Source words for the second batch: one inactive operation slot and the
second 1,024-cell scan chunk. -/
structure SecondPorts (assignment : Nat -> F) : Prop where
  constantWire : assignment 0 = 1
  step : Carries assignment
    (publicWord XOffset.step stepIndexBits) 1
  timestampIn : Carries assignment Compiler.timestampIn 1
  timestampOut : Carries assignment
    (publicWord XOffset.timestampOut Layout.timestampBits) 1
  pad : Carries assignment (operationPad wasm42x6 0) 1
  isWrite : Carries assignment (operationIsWrite wasm42x6 0) 0
  ram : Carries assignment (operationRam wasm42x6 0) 0
  address : Carries assignment (operationAddress wasm42x6 0) 0
  readValue : Carries assignment (operationReadValue wasm42x6 0) 0
  writeValue : Carries assignment (operationWriteValue wasm42x6 0) 0
  readTimestamp : Carries assignment
    (operationReadTimestamp wasm42x6 0) 0
  initialScanValue : forall slot, slot < 1024 ->
    Carries assignment (scanValue wasm42x6 false slot)
      (Memory.initialRamAt slot).value
  initialScanTimestamp : forall slot, slot < 1024 ->
    Carries assignment (scanTimestamp wasm42x6 false slot)
      (Memory.initialRamAt slot).timestamp
  finalScanValue : forall slot, slot < 1024 ->
    Carries assignment (scanValue wasm42x6 true slot)
      (Memory.finalRamAt slot).value
  finalScanTimestamp : forall slot, slot < 1024 ->
    Carries assignment (scanTimestamp wasm42x6 true slot)
      (Memory.finalRamAt slot).timestamp

private theorem entryOfFields_fieldOfNat
    (timestamp globalIndex value : Nat)
    (timestampBound : timestamp < goldilocksModulus)
    (indexBound : globalIndex < goldilocksModulus)
    (valueBound : value < goldilocksModulus) :
    entryOfFields
        (Compiler.fieldOfNat timestamp)
        (Compiler.fieldOfNat globalIndex)
        (Compiler.fieldOfNat value) =
      { timestamp := timestamp
        globalIndex := globalIndex
        value := value } := by
  simp [entryOfFields, Compiler.fieldOfNat,
    Nat.mod_eq_of_lt timestampBound,
    Nat.mod_eq_of_lt indexBound,
    Nat.mod_eq_of_lt valueBound]

private theorem scanGlobalIndex_carries
    (assignment : Nat -> F) (step : Nat)
    (constantWire : assignment 0 = 1)
    (stepCarries : Carries assignment
      (publicWord XOffset.step stepIndexBits) step)
    (slot : Nat) :
    Carries assignment (scanGlobalIndex wasm42x6 slot)
      (1024 * step + slot) := by
  unfold Carries fieldValue at stepCarries ⊢
  simp only [scanGlobalIndex, Rows.LinearCombination.eval_add,
    Rows.LinearCombination.eval_scale,
    Rows.LinearCombination.eval_constant]
  rw [stepCarries, constantWire, Fin.mul_one]
  rw [← fieldOfNat_mul, ← fieldOfNat_add]
  rfl

private theorem operationGlobalIndex_first
    (assignment : Nat -> F) (ports : FirstPorts assignment) :
    Carries assignment (operationGlobalIndex wasm42x6 0)
      Memory.loadedGlobalIndex := by
  have address := ports.address
  have ram := ports.ram
  unfold Carries fieldValue at address ram ⊢
  simp only [operationGlobalIndex, Rows.LinearCombination.eval_add,
    Rows.LinearCombination.eval_scale]
  rw [address, ram, ← fieldOfNat_mul, ← fieldOfNat_add]
  rfl

private theorem firstCount
    (assignment : Nat -> F)
    (ports : FirstPorts assignment)
    (accepted : Accepted assignment wasm42x6) :
    Carries assignment (operationCountWord wasm42x6 0) 1 := by
  have count := (accepted.operations 0 (by decide)).countExact
  have pad := ports.pad
  unfold LinearEqual at count
  unfold Carries fieldValue at pad ⊢
  simpa [Rows.LinearCombination.eval_add,
    Rows.LinearCombination.eval_sub,
    Rows.LinearCombination.eval_zero,
    Rows.LinearCombination.eval_constant, one,
    ports.constantWire, pad] using count

private theorem firstWriteTimestamp
    (assignment : Nat -> F)
    (ports : FirstPorts assignment)
    (accepted : Accepted assignment wasm42x6) :
    Carries assignment (operationWriteTimestamp wasm42x6 0) 1 := by
  have timestampInExact := ports.timestampIn
  unfold Carries fieldValue at timestampInExact ⊢
  simp only [operationWriteTimestamp, Rows.LinearCombination.eval_add]
  rw [timestampInExact]
  have count := firstCount assignment ports accepted
  unfold Carries fieldValue at count
  rw [count, ← fieldOfNat_add]

theorem first_operation_entries
    (assignment : Nat -> F)
    (ports : FirstPorts assignment)
    (satisfied : Satisfies (rows wasm42x6) assignment) :
    operationEntry assignment wasm42x6 0 false = Memory.readCell ∧
      operationEntry assignment wasm42x6 0 true = Memory.writeCell := by
  have accepted := accepted_of_rows assignment wasm42x6
    ports.constantWire satisfied
  have global := operationGlobalIndex_first assignment ports
  have writeTimestamp := firstWriteTimestamp assignment ports accepted
  unfold operationEntry
  constructor
  · simp only [Bool.false_eq_true, ↓reduceIte]
    rw [show fieldValue assignment (operationReadTimestamp wasm42x6 0) =
        Compiler.fieldOfNat 0 by exact ports.readTimestamp]
    rw [global, ports.readValue]
    exact entryOfFields_fieldOfNat 0 Memory.loadedGlobalIndex 42
      (by decide) (by decide) (by decide)
  · simp only [↓reduceIte]
    rw [show fieldValue assignment (operationWriteTimestamp wasm42x6 0) =
        Compiler.fieldOfNat 1 by exact writeTimestamp]
    rw [global, ports.writeValue]
    exact entryOfFields_fieldOfNat 1 Memory.loadedGlobalIndex 42
      (by decide) (by decide) (by decide)

private theorem scanEntry_eq
    (assignment : Nat -> F) (final : Bool) (step slot : Nat)
    (constantWire : assignment 0 = 1)
    (stepCarries : Carries assignment
      (publicWord XOffset.step stepIndexBits) step)
    (expected : MemTuple)
    (valueCarries : Carries assignment
      (scanValue wasm42x6 final slot) expected.value)
    (timestampCarries : Carries assignment
      (scanTimestamp wasm42x6 final slot) expected.timestamp)
    (indexExact : expected.globalIndex = 1024 * step + slot)
    (timestampBound : expected.timestamp < goldilocksModulus)
    (indexBound : expected.globalIndex < goldilocksModulus)
    (valueBound : expected.value < goldilocksModulus) :
    scanEntry assignment wasm42x6 final slot = expected := by
  unfold scanEntry
  have global := scanGlobalIndex_carries assignment step constantWire
    stepCarries slot
  unfold Carries at valueCarries timestampCarries global
  rw [valueCarries, timestampCarries, global, ← indexExact]
  simpa using entryOfFields_fieldOfNat expected.timestamp
    expected.globalIndex expected.value timestampBound indexBound valueBound

theorem first_scan_entries
    (assignment : Nat -> F) (ports : FirstPorts assignment) :
    scanEntries assignment wasm42x6 false 1024 = Memory.romChunk ∧
      scanEntries assignment wasm42x6 true 1024 = Memory.romChunk := by
  have modulusPositive : 0 < goldilocksModulus := by decide
  have slotsFit : 1024 < goldilocksModulus := by decide
  rw [scanEntries_eq_map_range, scanEntries_eq_map_range]
  constructor <;>
    apply List.map_congr_left <;>
    intro slot slotMember <;>
    have slotBound : slot < 1024 := List.mem_range.mp slotMember
  · exact scanEntry_eq assignment false 0 slot ports.constantWire
      ports.step (initialRomAt slot)
      (ports.initialScanValue slot slotBound)
      (ports.initialScanTimestamp slot slotBound)
      (by simp [initialRomAt, Memory.blankCell])
      (by simpa [initialRomAt, Memory.blankCell] using modulusPositive)
      (by simp [initialRomAt, Memory.blankCell]; omega)
      (by simpa [initialRomAt, Memory.blankCell] using modulusPositive)
  · exact scanEntry_eq assignment true 0 slot ports.constantWire
      ports.step (finalRomAt slot)
      (ports.finalScanValue slot slotBound)
      (ports.finalScanTimestamp slot slotBound)
      (by simp [finalRomAt, Memory.blankCell])
      (by simpa [finalRomAt, Memory.blankCell] using modulusPositive)
      (by simp [finalRomAt, Memory.blankCell]; omega)
      (by simpa [finalRomAt, Memory.blankCell] using modulusPositive)

theorem first_activation
    (assignment : Nat -> F) (ports : FirstPorts assignment) :
    ActivationMatches assignment wasm42x6 (fun _ => true) 1 := by
  intro slot slotBound
  have slotZero : slot = 0 := by omega
  subst slot
  simpa [Carries, fieldOfNat_zero] using ports.pad

theorem second_activation
    (assignment : Nat -> F) (ports : SecondPorts assignment) :
    ActivationMatches assignment wasm42x6 (fun _ => false) 1 := by
  intro slot slotBound
  have slotZero : slot = 0 := by omega
  subst slot
  simpa [Carries, fieldOfNat_one] using ports.pad

private theorem initialRamAt_index (slot : Nat) :
    (Memory.initialRamAt slot).globalIndex = 1024 + slot := by
  by_cases slotZero : slot = 0
  · subst slot
    rfl
  · simp [Memory.initialRamAt, slotZero, Memory.blankCell,
      Memory.loadedGlobalIndex, Memory.romCells]

private theorem finalRamAt_index (slot : Nat) :
    (Memory.finalRamAt slot).globalIndex = 1024 + slot := by
  by_cases slotZero : slot = 0
  · subst slot
    rfl
  · simp [Memory.finalRamAt, slotZero, Memory.blankCell,
      Memory.loadedGlobalIndex, Memory.romCells]

private theorem initialRamAt_timestamp_bound (slot : Nat) :
    (Memory.initialRamAt slot).timestamp < goldilocksModulus := by
  by_cases slotZero : slot = 0 <;>
    simp [Memory.initialRamAt, slotZero, Memory.readCell,
      Memory.blankCell] <;> decide

private theorem finalRamAt_timestamp_bound (slot : Nat) :
    (Memory.finalRamAt slot).timestamp < goldilocksModulus := by
  by_cases slotZero : slot = 0 <;>
    simp [Memory.finalRamAt, slotZero, Memory.writeCell,
      Memory.blankCell] <;> decide

private theorem initialRamAt_value_bound (slot : Nat) :
    (Memory.initialRamAt slot).value < goldilocksModulus := by
  by_cases slotZero : slot = 0 <;>
    simp [Memory.initialRamAt, slotZero, Memory.readCell,
      Memory.blankCell] <;> decide

private theorem finalRamAt_value_bound (slot : Nat) :
    (Memory.finalRamAt slot).value < goldilocksModulus := by
  by_cases slotZero : slot = 0 <;>
    simp [Memory.finalRamAt, slotZero, Memory.writeCell,
      Memory.blankCell] <;> decide

theorem second_scan_entries
    (assignment : Nat -> F) (ports : SecondPorts assignment) :
    scanEntries assignment wasm42x6 false 1024 =
        Memory.initialRamChunk ∧
      scanEntries assignment wasm42x6 true 1024 =
        Memory.finalRamChunk := by
  have slotsFit : 2048 < goldilocksModulus := by decide
  rw [scanEntries_eq_map_range, scanEntries_eq_map_range,
    Memory.initialRamChunk_eq_map, Memory.finalRamChunk_eq_map]
  constructor <;>
    apply List.map_congr_left <;>
    intro slot slotMember <;>
    have slotBound : slot < 1024 := List.mem_range.mp slotMember
  · exact scanEntry_eq assignment false 1 slot ports.constantWire
      ports.step (Memory.initialRamAt slot)
      (ports.initialScanValue slot slotBound)
      (ports.initialScanTimestamp slot slotBound)
      (by rw [initialRamAt_index])
      (initialRamAt_timestamp_bound slot)
      (by rw [initialRamAt_index]; omega)
      (initialRamAt_value_bound slot)
  · exact scanEntry_eq assignment true 1 slot ports.constantWire
      ports.step (Memory.finalRamAt slot)
      (ports.finalScanValue slot slotBound)
      (ports.finalScanTimestamp slot slotBound)
      (by rw [finalRamAt_index])
      (finalRamAt_timestamp_bound slot)
      (by rw [finalRamAt_index]; omega)
      (finalRamAt_value_bound slot)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaBinding
