import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessWords
import Nightstream.Implementation.Lowering.Nebula.SourceProducts

/-!
Honest physical execution of the modular 42-times-6 Nebula relation.

Assurance tier: model-level.

This file constructs the declarative accepted record from the authoritative
benchmark memory trace and derives satisfaction of the exact numbered rows.
It does not compose the fifteen Nebula matrices with the four F-prime
matrices, select a terminal decider, export Rust, or prove challenge security.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonest

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceProducts
open Nightstream.Implementation.Lowering.Nebula.BitWitness
open Nightstream.Implementation.Lowering.Nebula.BitSemantics
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessAccess
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessWords
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

private theorem isBit_of_bitField
    (assignment : Nat -> F) (column value bit : Nat)
    (exact : assignment column = bitField value bit) :
    IsBit assignment column := by
  unfold IsBit
  rw [exact]
  unfold bitField
  split <;> rfl

private theorem isBit_of_zero_or_one
    (assignment : Nat -> F) (column : Nat)
    (root : assignment column = 0 ∨ assignment column = 1) :
    IsBit assignment column := by
  unfold IsBit
  rcases root with zero | one
  · simp [zero]
  · simp [one]

private theorem cellBit_zero_or_one (cell : MemTuple) (offset : Nat) :
    cellBit cell offset = 0 ∨ cellBit cell offset = 1 := by
  unfold cellBit
  split <;> exact bitField_zero_or_one _ _

private theorem operation_lane_isBit
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < wasm42x6.operationBits) :
    IsBit (assignment challengeValues batch)
      (wasm42x6.operationSlot 0 + offset) := by
  apply isBit_of_zero_or_one
  rw [show wasm42x6.operationSlot 0 + offset =
      wasm42x6.operationLane + offset by rfl,
    assignment_operation challengeValues batch offset offsetBound]
  unfold operationBit
  by_cases first : offset = 0
  · rw [if_pos first]
    exact bitField_zero_or_one _ _
  rw [if_neg first]
  by_cases second : offset = 1
  · rw [if_pos second]
    exact bitField_zero_or_one _ _
  rw [if_neg second]
  by_cases third : offset = 2
  · rw [if_pos third]
    exact bitField_zero_or_one _ _
  rw [if_neg third]
  by_cases address : offset < 3 + wasm42x6.addressBits
  · rw [if_pos address]
    exact bitField_zero_or_one _ _
  rw [if_neg address]
  by_cases readValue : offset < 3 + wasm42x6.addressBits + valueBits
  · rw [if_pos readValue]
    exact bitField_zero_or_one _ _
  rw [if_neg readValue]
  by_cases writeValue :
      offset < 3 + wasm42x6.addressBits + 2 * valueBits
  · rw [if_pos writeValue]
    exact bitField_zero_or_one _ _
  rw [if_neg writeValue]
  exact bitField_zero_or_one _ _

private theorem operation_auxiliary_isBit
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < wasm42x6.operationAuxiliaryBits) :
    IsBit (assignment challengeValues batch)
      (wasm42x6.auxiliaryStart + offset) := by
  apply isBit_of_zero_or_one
  rw [assignment_operation_auxiliary challengeValues batch offset offsetBound]
  unfold operationAuxiliaryBit
  by_cases diff : offset < Layout.timestampBits
  · rw [if_pos diff]
    exact bitField_zero_or_one _ _
  rw [if_neg diff]
  by_cases count : offset < Layout.timestampBits + wasm42x6.countBits
  · rw [if_pos count]
    exact bitField_zero_or_one _ _
  rw [if_neg count]
  by_cases readProduct :
      offset < Layout.timestampBits + wasm42x6.countBits + extensionBits
  · rw [if_pos readProduct]
    unfold kBit
    exact bitField_zero_or_one _ _
  rw [if_neg readProduct]
  unfold kBit
  exact bitField_zero_or_one _ _

private theorem scan_cell_isBit
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (slot offset : Nat)
    (slotBound : slot < wasm42x6.scanSlots)
    (offsetBound : offset < cellBits) :
    IsBit (assignment challengeValues batch)
      (scanCellStart wasm42x6 final slot + offset) := by
  apply isBit_of_zero_or_one
  cases final
  · rw [show scanCellStart wasm42x6 false slot + offset =
        wasm42x6.initialScanLane + slot * cellBits + offset by rfl,
      assignment_initial_cell challengeValues batch slot offset slotBound
        offsetBound]
    exact cellBit_zero_or_one _ _
  · rw [show scanCellStart wasm42x6 true slot + offset =
        wasm42x6.finalScanLane + slot * cellBits + offset by rfl,
      assignment_final_cell challengeValues batch slot offset slotBound
        offsetBound]
    exact cellBit_zero_or_one _ _

private theorem scan_auxiliary_isBit
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat)
    (offsetBound : offset < wasm42x6.scanSlots * scanAuxiliaryBits) :
    IsBit (assignment challengeValues batch)
      (wasm42x6.scanAuxiliaryStart + offset) := by
  apply isBit_of_zero_or_one
  rw [assignment_scan_auxiliary challengeValues batch offset offsetBound]
  unfold scanAuxiliaryBit
  dsimp only
  by_cases initial : offset % scanAuxiliaryBits < extensionBits
  · rw [if_pos initial]
    unfold kBit
    exact bitField_zero_or_one _ _
  rw [if_neg initial]
  unfold kBit
  exact bitField_zero_or_one _ _

private theorem operation_bits
    (challengeValues : Challenges) (batch : Fin 2) :
    (∀ offset, offset < wasm42x6.operationBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationSlot 0 + offset)) ∧
    (∀ offset, offset < Layout.timestampBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationDiff 0 + offset)) ∧
    (∀ offset, offset < wasm42x6.countBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationCount 0 + offset)) ∧
    (∀ offset, offset < extensionBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationReadProduct 0 + offset)) ∧
    (∀ offset, offset < extensionBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationWriteProduct 0 + offset)) := by
  refine ⟨operation_lane_isBit challengeValues batch, ?_, ?_, ?_, ?_⟩
  · intro offset offsetBound
    have bit := operation_auxiliary_isBit challengeValues batch offset (by
      unfold Params.operationAuxiliaryBits
      omega)
    simpa only [Params.operationDiff, Params.operationAuxiliary, Nat.zero_mul,
      Nat.zero_add, Nat.add_assoc] using bit
  · intro offset offsetBound
    have bit := operation_auxiliary_isBit challengeValues batch
      (Layout.timestampBits + offset) (by
        unfold Params.operationAuxiliaryBits
        omega)
    simpa only [Params.operationCount, Params.operationDiff,
      Params.operationAuxiliary, Nat.zero_mul, Nat.zero_add, Nat.add_assoc]
      using bit
  · intro offset offsetBound
    have bit := operation_auxiliary_isBit challengeValues batch
      (Layout.timestampBits + wasm42x6.countBits + offset) (by
        unfold Params.operationAuxiliaryBits
        omega)
    simpa only [Params.operationReadProduct, Params.operationCount,
      Params.operationDiff, Params.operationAuxiliary, Nat.zero_mul,
      Nat.zero_add, Nat.add_assoc] using bit
  · intro offset offsetBound
    have bit := operation_auxiliary_isBit challengeValues batch
      (Layout.timestampBits + wasm42x6.countBits + extensionBits + offset) (by
        unfold Params.operationAuxiliaryBits
        omega)
    simpa only [Params.operationWriteProduct, Params.operationReadProduct,
      Params.operationCount, Params.operationDiff, Params.operationAuxiliary,
      Nat.zero_mul, Nat.zero_add, Nat.add_assoc] using bit

private theorem scan_bits
    (challengeValues : Challenges) (batch : Fin 2)
    (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    (∀ offset, offset < cellBits ->
      IsBit (assignment challengeValues batch)
        (scanCellStart wasm42x6 false slot + offset)) ∧
    (∀ offset, offset < extensionBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.initialScanProduct slot + offset)) ∧
    (∀ offset, offset < cellBits ->
      IsBit (assignment challengeValues batch)
        (scanCellStart wasm42x6 true slot + offset)) ∧
    (∀ offset, offset < extensionBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.finalScanProduct slot + offset)) := by
  refine ⟨fun offset offsetBound =>
      scan_cell_isBit challengeValues batch false slot offset slotBound
        offsetBound,
    ?_, fun offset offsetBound =>
      scan_cell_isBit challengeValues batch true slot offset slotBound
        offsetBound, ?_⟩
  · intro offset offsetBound
    have bit := scan_auxiliary_isBit challengeValues batch
      (slot * scanAuxiliaryBits + offset) (by
        change slot * 256 + offset < 1024 * 256
        change slot < 1024 at slotBound
        change offset < 128 at offsetBound
        omega)
    simpa only [Params.initialScanProduct, Nat.add_assoc] using bit
  · intro offset offsetBound
    have bit := scan_auxiliary_isBit challengeValues batch
      (slot * scanAuxiliaryBits + extensionBits + offset) (by
        change slot * 256 + 128 + offset < 1024 * 256
        change slot < 1024 at slotBound
        change offset < 128 at offsetBound
        omega)
    simpa only [Params.finalScanProduct, Params.initialScanProduct,
      Nat.add_assoc] using bit

theorem physical_bits_and_fillers
    (challengeValues : Challenges) (batch : Fin 2) :
    (∀ column, column ∈ wasm42x6.fillerColumns ->
      assignment challengeValues batch column = 0) ∧
    (∀ offset, offset < wasm42x6.operationBits ->
      IsBit (assignment challengeValues batch)
        (wasm42x6.operationSlot 0 + offset)) ∧
    (∀ slot, slot < wasm42x6.scanSlots ->
      ∀ offset, offset < cellBits ->
        IsBit (assignment challengeValues batch)
          (scanCellStart wasm42x6 false slot + offset)) := by
  exact ⟨filler_get challengeValues batch,
    (operation_bits challengeValues batch).1,
    fun slot slotBound => (scan_bits challengeValues batch slot slotBound).1⟩

private theorem previous_operation_product_exact
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool) :
    previousOperationProductValue (assignment challengeValues batch)
        wasm42x6 0 write =
      inputProductAt challengeValues batch (if write then 1 else 0) := by
  cases write
  · simpa [previousOperationProductValue, previousOperationProduct,
      inputProduct] using input_product_exact challengeValues batch 0 (by decide)
  · simpa [previousOperationProductValue, previousOperationProduct,
      inputProduct] using input_product_exact challengeValues batch 1 (by decide)

private theorem batch_val_zero_of_not_second
    (batch : Fin 2) (second : isSecond batch = false) : batch.val = 0 := by
  have bound := batch.isLt
  unfold isSecond at second
  simp only [decide_eq_false_iff_not] at second
  omega

private theorem entryOfFields_fieldOfNat
    (timestamp globalIndex value : Nat)
    (timestampBound : timestamp < goldilocksModulus)
    (indexBound : globalIndex < goldilocksModulus)
    (valueBound : value < goldilocksModulus) :
    entryOfFields
        (fieldOfNat timestamp) (fieldOfNat globalIndex) (fieldOfNat value) =
      { timestamp := timestamp, globalIndex := globalIndex, value := value } := by
  simp [entryOfFields, fieldOfNat, Nat.mod_eq_of_lt timestampBound,
    Nat.mod_eq_of_lt indexBound, Nat.mod_eq_of_lt valueBound]

private theorem operation_global_index_active
    (challengeValues : Challenges) (batch : Fin 2)
    (second : isSecond batch = false) :
    fieldValue (assignment challengeValues batch)
        (operationGlobalIndex wasm42x6 0) =
      fieldOfNat Memory.loadedGlobalIndex := by
  have address := operation_address_exact challengeValues batch
  have ram := operation_scalar_exact challengeValues batch 2
  unfold fieldValue at address ram ⊢
  simp only [operationGlobalIndex, Rows.LinearCombination.eval_add,
    Rows.LinearCombination.eval_scale]
  rw [address, ram]
  simp only [second, Bool.false_eq_true, ↓reduceIte, fieldOfNat_zero,
    Fin.zero_add, ← fieldOfNat_mul, ← fieldOfNat_add]
  rfl

private theorem operation_write_timestamp_active
    (challengeValues : Challenges) (batch : Fin 2)
    (second : isSecond batch = false) :
    fieldValue (assignment challengeValues batch)
        (operationWriteTimestamp wasm42x6 0) = fieldOfNat 1 := by
  have batchZero := batch_val_zero_of_not_second batch second
  unfold fieldValue operationWriteTimestamp
  rw [Rows.LinearCombination.eval_add, timestamp_in_word_exact,
    operation_count_exact, batchZero]
  simp [second, ← fieldOfNat_add]

private theorem operation_entry_active
    (challengeValues : Challenges) (batch : Fin 2)
    (second : isSecond batch = false) (write : Bool) :
    operationEntry (assignment challengeValues batch) wasm42x6 0 write =
      if write then Memory.writeCell else Memory.readCell := by
  have global := operation_global_index_active challengeValues batch second
  unfold fieldValue at global
  cases write
  · simp only [Bool.false_eq_true, ↓reduceIte]
    unfold operationEntry fieldValue
    simp only [Bool.false_eq_true, ↓reduceIte]
    rw [operation_read_timestamp_exact, global,
      operation_read_value_exact]
    simp only [second, Bool.false_eq_true, ↓reduceIte]
    exact entryOfFields_fieldOfNat 0 Memory.loadedGlobalIndex 42
      (by decide) (by decide) (by decide)
  · simp only [↓reduceIte]
    have writeTimestamp :=
      operation_write_timestamp_active challengeValues batch second
    unfold fieldValue at writeTimestamp
    unfold operationEntry fieldValue
    simp only [↓reduceIte]
    rw [writeTimestamp, global, operation_write_value_exact]
    simp only [second, Bool.false_eq_true, ↓reduceIte]
    exact entryOfFields_fieldOfNat 1 Memory.loadedGlobalIndex 42
      (by decide) (by decide) (by decide)

private theorem operation_gate_exact
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool) :
    operationGate (assignment challengeValues batch) wasm42x6 0 write =
      if isSecond batch then K.one
      else fingerprint challengeValues
        (if write then Memory.writeCell else Memory.readCell) := by
  cases second : isSecond batch
  · have padZero :
        fieldValue (assignment challengeValues batch)
            (operationPad wasm42x6 0) = 0 := by
      simpa [fieldValue, second] using
        operation_scalar_exact challengeValues batch 0
    rw [operationGate_eq_fingerprint_of_active
      (assignment challengeValues batch) wasm42x6 0 write
      (assignment_constant challengeValues batch) padZero,
      challenges_exact]
    rw [operation_entry_active challengeValues batch second write]
    simp
  · have padOne :
        fieldValue (assignment challengeValues batch)
            (operationPad wasm42x6 0) = 1 := by
      simpa [fieldValue, second] using
        operation_scalar_exact challengeValues batch 0
    rw [operationGate_eq_one_of_inactive
      (assignment challengeValues batch) wasm42x6 0 write
      (assignment_constant challengeValues batch) padOne]
    simp

private theorem operation_product_recurrence
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool) :
    operationProduct (assignment challengeValues batch) wasm42x6 0 write =
      K.mul
        (previousOperationProductValue (assignment challengeValues batch)
          wasm42x6 0 write)
        (operationGate (assignment challengeValues batch) wasm42x6 0 write) := by
  rw [operation_product_exact, previous_operation_product_exact,
    operation_gate_exact]
  unfold operationProductAt
  cases second : isSecond batch <;> simp [second]

private theorem operation_accepted
    (challengeValues : Challenges) (batch : Fin 2) :
    OperationAccepted (assignment challengeValues batch) wasm42x6 0 := by
  obtain ⟨laneBits, diffBits, countBits, readProductBits,
      writeProductBits⟩ := operation_bits challengeValues batch
  have padExact := operation_scalar_exact challengeValues batch 0
  have isWriteExact := operation_scalar_exact challengeValues batch 1
  have ramExact := operation_scalar_exact challengeValues batch 2
  refine {
    laneBits := laneBits
    diffBits := diffBits
    countBits := countBits
    readProductBits := readProductBits
    writeProductBits := writeProductBits
    countExact := ?_
    readWritesBack := ?_
    timestampOrdered := ?_
    romReadOnly := ?_
    romAddressRange := ?_
    padIsWriteZero := ?_
    padRamZero := ?_
    padAddressZero := ?_
    padReadValueZero := ?_
    padWriteValueZero := ?_
    padReadTimestampZero := ?_
    readProductExact := operation_product_recurrence challengeValues batch false
    writeProductExact := operation_product_recurrence challengeValues batch true }
  · unfold LinearEqual
    simp only [Rows.LinearCombination.eval_add,
      Rows.LinearCombination.eval_zero, Rows.LinearCombination.eval_sub,
      Rows.LinearCombination.eval_constant, one,
      assignment_constant]
    rw [operation_count_exact, padExact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    simp only [Rows.LinearCombination.eval_sub,
      Rows.LinearCombination.eval_constant, one, assignment_constant]
    rw [isWriteExact, operation_write_value_exact,
      operation_read_value_exact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    simp only [Rows.LinearCombination.eval_sub,
      Rows.LinearCombination.eval_constant, one, assignment_constant,
      operationWriteTimestamp, Rows.LinearCombination.eval_add]
    rw [padExact, timestamp_in_word_exact, operation_count_exact,
      operation_read_timestamp_exact, operation_diff_exact]
    cases second : isSecond batch
    · rw [batch_val_zero_of_not_second batch second]
      simp [second]
    · simp [second]
  · unfold ProductZero
    simp only [Rows.LinearCombination.eval_sub,
      Rows.LinearCombination.eval_constant, one, assignment_constant]
    rw [isWriteExact, ramExact]
    cases second : isSecond batch <;> simp [second]
  · intro offset offsetBound
    have impossible : offset < 0 := by
      simpa [wasm42x6, Params.addressBits] using offsetBound
    omega
  · unfold ProductZero
    rw [padExact, isWriteExact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    rw [padExact, ramExact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    rw [padExact, operation_address_exact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    rw [padExact, operation_read_value_exact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    rw [padExact, operation_write_value_exact]
    cases second : isSecond batch <;> simp [second]
  · unfold ProductZero
    rw [padExact, operation_read_timestamp_exact]
    cases second : isSecond batch <;> simp [second]

private theorem batch_val_one_of_second
    (batch : Fin 2) (second : isSecond batch = true) : batch.val = 1 := by
  simpa [isSecond] using second

private theorem cellAt_global_index
    (batch : Fin 2) (final : Bool) (slot : Nat) :
    (cellAt batch final slot).globalIndex = 1024 * batch.val + slot := by
  cases second : isSecond batch
  · have batchZero := batch_val_zero_of_not_second batch second
    simp [cellAt, firstCell, second, Memory.blankCell, batchZero]
  · have batchOne := batch_val_one_of_second batch second
    cases final <;> by_cases slotZero : slot = 0 <;>
      simp [cellAt, secondCell, second, slotZero, Memory.initialRamAt,
        Memory.finalRamAt, Memory.readCell, Memory.writeCell,
        Memory.blankCell, Memory.loadedGlobalIndex, Memory.romCells,
        batchOne] <;> omega

private theorem scan_global_index_exact
    (challengeValues : Challenges) (batch : Fin 2) (slot : Nat) :
    fieldValue (assignment challengeValues batch)
        (scanGlobalIndex wasm42x6 slot) =
      fieldOfNat (1024 * batch.val + slot) := by
  unfold fieldValue scanGlobalIndex
  rw [Rows.LinearCombination.eval_add, Rows.LinearCombination.eval_scale,
    Rows.LinearCombination.eval_constant, step_word_exact,
    assignment_constant, Fin.mul_one, ← fieldOfNat_mul,
    ← fieldOfNat_add]
  rfl

private theorem cellAt_timestamp_bound
    (batch : Fin 2) (final : Bool) (slot : Nat) :
    (cellAt batch final slot).timestamp < goldilocksModulus := by
  cases final <;> by_cases second : isSecond batch = true <;>
    by_cases slotZero : slot = 0 <;>
    simp [cellAt, firstCell, secondCell, second, slotZero,
      Memory.initialRamAt, Memory.finalRamAt, Memory.readCell,
      Memory.writeCell, Memory.blankCell] <;> decide

private theorem cellAt_value_bound
    (batch : Fin 2) (final : Bool) (slot : Nat) :
    (cellAt batch final slot).value < goldilocksModulus := by
  cases final <;> by_cases second : isSecond batch = true <;>
    by_cases slotZero : slot = 0 <;>
    simp [cellAt, firstCell, secondCell, second, slotZero,
      Memory.initialRamAt, Memory.finalRamAt, Memory.readCell,
      Memory.writeCell, Memory.blankCell] <;> decide

private theorem scan_entry_exact
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) (slotBound : slot < 1024) :
    scanEntry (assignment challengeValues batch) wasm42x6 final slot =
      cellAt batch final slot := by
  have global := scan_global_index_exact challengeValues batch slot
  unfold fieldValue at global
  unfold scanEntry fieldValue
  rw [scan_timestamp_exact challengeValues batch final slot slotBound,
    global, scan_value_exact challengeValues batch final slot slotBound,
    ← cellAt_global_index batch final slot]
  exact entryOfFields_fieldOfNat
    (cellAt batch final slot).timestamp
    (cellAt batch final slot).globalIndex
    (cellAt batch final slot).value
    (cellAt_timestamp_bound batch final slot)
    (by
      rw [cellAt_global_index]
      have batchBound := batch.isLt
      have small : 1024 * batch.val + slot < 2048 := by omega
      exact Nat.lt_trans small (by decide))
    (cellAt_value_bound batch final slot)

private theorem scan_factor_exact
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) (slotBound : slot < 1024) :
    scanFactor (assignment challengeValues batch) wasm42x6 final slot =
      fingerprint challengeValues (cellAt batch final slot) := by
  rw [scanFactor_eq_fingerprint, challenges_exact,
    scan_entry_exact challengeValues batch final slot slotBound]

private theorem previous_scan_product_exact
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) (slotBound : slot < 1024) :
    previousScanProductValue (assignment challengeValues batch)
        wasm42x6 final slot =
      if slot = 0 then
        inputProductAt challengeValues batch (if final then 3 else 2)
      else scanProductAt challengeValues batch final slot := by
  cases slot with
  | zero =>
      cases final
      · simpa [previousScanProductValue, previousScanProduct,
          inputProduct] using
          input_product_exact challengeValues batch 2 (by decide)
      · simpa [previousScanProductValue, previousScanProduct,
          inputProduct] using
          input_product_exact challengeValues batch 3 (by decide)
  | succ prior =>
      have priorBound : prior < 1024 := by omega
      simpa [previousScanProductValue, previousScanProduct] using
        scan_product_exact challengeValues batch final prior priorBound

private theorem scanProductAt_recurrence
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) :
    scanProductAt challengeValues batch final (slot + 1) =
      K.mul
        (if slot = 0 then
          inputProductAt challengeValues batch (if final then 3 else 2)
        else scanProductAt challengeValues batch final slot)
        (fingerprint challengeValues (cellAt batch final slot)) := by
  cases slot with
  | zero =>
      simp only [scanProductAt, productPrefix]
      rw [if_pos True.intro]
      exact congrArg
        (fun value => K.mul
          (inputProductAt challengeValues batch (if final then 3 else 2))
          value)
        (extensionLaws.one_mul _)
  | succ prior =>
      simp only [scanProductAt, productPrefix, Nat.succ_ne_zero, if_false]
      exact (extensionLaws.mul_assoc _ _ _).symm

private theorem scan_product_recurrence
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) (slotBound : slot < 1024) :
    scanProduct (assignment challengeValues batch) wasm42x6 final slot =
      K.mul
        (previousScanProductValue (assignment challengeValues batch)
          wasm42x6 final slot)
        (scanFactor (assignment challengeValues batch) wasm42x6 final slot) := by
  rw [scan_product_exact challengeValues batch final slot slotBound,
    previous_scan_product_exact challengeValues batch final slot slotBound,
    scan_factor_exact challengeValues batch final slot slotBound]
  exact scanProductAt_recurrence challengeValues batch final slot

private theorem scan_accepted
    (challengeValues : Challenges) (batch : Fin 2)
    (slot : Nat) (slotBound : slot < wasm42x6.scanSlots) :
    ScanAccepted (assignment challengeValues batch) wasm42x6 slot := by
  have concreteBound : slot < 1024 := by
    simpa only [wasm42x6_scanSlots] using slotBound
  obtain ⟨initialCellBits, initialProductBits, finalCellBits,
      finalProductBits⟩ := scan_bits challengeValues batch slot slotBound
  exact {
    initialCellBits := initialCellBits
    initialProductBits := initialProductBits
    finalCellBits := finalCellBits
    finalProductBits := finalProductBits
    initialProductExact := scan_product_recurrence challengeValues batch false
      slot concreteBound
    finalProductExact := scan_product_recurrence challengeValues batch true
      slot concreteBound }

private theorem boundary_accepted
    (challengeValues : Challenges) (batch : Fin 2) :
    BoundaryAccepted (assignment challengeValues batch) wasm42x6 := by
  refine {
    timestampExact := ?_
    product0 := ?_
    product1 := ?_
    product2 := ?_
    product3 := ?_ }
  · unfold LinearEqual
    rw [Rows.LinearCombination.eval_add, timestamp_out_word_exact,
      timestamp_in_word_exact]
    change fieldOfNat 1 =
      fieldOfNat batch.val +
        Rows.LinearCombination.eval (assignment challengeValues batch)
          (operationCountWord wasm42x6 0)
    rw [operation_count_exact]
    cases second : isSecond batch
    · rw [batch_val_zero_of_not_second batch second]
      simp [second]
    · rw [batch_val_one_of_second batch second]
      simp [second]
  · calc
      outputProduct (assignment challengeValues batch) 0 =
          outputProductAt challengeValues batch 0 :=
        output_product_exact challengeValues batch 0 (by decide)
      _ = operationProduct (assignment challengeValues batch)
          wasm42x6 0 false :=
        (operation_product_exact challengeValues batch false).symm
      _ = boundaryProductValue (assignment challengeValues batch)
          wasm42x6 0 := by rfl
  · calc
      outputProduct (assignment challengeValues batch) 1 =
          outputProductAt challengeValues batch 1 :=
        output_product_exact challengeValues batch 1 (by decide)
      _ = operationProduct (assignment challengeValues batch)
          wasm42x6 0 true :=
        (operation_product_exact challengeValues batch true).symm
      _ = boundaryProductValue (assignment challengeValues batch)
          wasm42x6 1 := by rfl
  · calc
      outputProduct (assignment challengeValues batch) 2 =
          outputProductAt challengeValues batch 2 :=
        output_product_exact challengeValues batch 2 (by decide)
      _ = scanProduct (assignment challengeValues batch)
          wasm42x6 false 1023 := by
        simpa using (scan_product_exact challengeValues batch false 1023
          (by decide)).symm
      _ = boundaryProductValue (assignment challengeValues batch)
          wasm42x6 2 := by rfl
  · calc
      outputProduct (assignment challengeValues batch) 3 =
          outputProductAt challengeValues batch 3 :=
        output_product_exact challengeValues batch 3 (by decide)
      _ = scanProduct (assignment challengeValues batch)
          wasm42x6 true 1023 := by
        simpa using (scan_product_exact challengeValues batch true 1023
          (by decide)).symm
      _ = boundaryProductValue (assignment challengeValues batch)
          wasm42x6 3 := by rfl

/-- The benchmark witness is accepted by the exact declarative relation for
either of its two batches. -/
theorem accepted
    (challengeValues : Challenges) (batch : Fin 2) :
    Accepted (assignment challengeValues batch) wasm42x6 := {
  constantWire := assignment_constant challengeValues batch
  fillerZero := filler_get challengeValues batch
  operations := by
    intro slot slotBound
    have slotZero : slot = 0 := by
      have lessOne : slot < 1 := by
        simpa only [wasm42x6_operationSlots] using slotBound
      omega
    subst slot
    exact operation_accepted challengeValues batch
  scans := scan_accepted challengeValues batch
  boundary := boundary_accepted challengeValues batch
}

/-- Row-level honest completeness for the exact numbered 422,465-row
program. The row count is derived by the compiler; it is not a premise. -/
theorem rows_satisfied
    (challengeValues : Challenges) (batch : Fin 2) :
    Satisfies (rows wasm42x6) (assignment challengeValues batch) :=
  rows_honest_of_accepted (assignment challengeValues batch) wasm42x6
    (accepted challengeValues batch)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonest
