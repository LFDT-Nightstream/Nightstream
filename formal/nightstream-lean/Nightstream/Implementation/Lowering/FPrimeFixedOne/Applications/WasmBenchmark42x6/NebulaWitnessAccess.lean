import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout

/-!
Exact access lemmas for the modular 42-times-6 Nebula witness.

Assurance tier: model-level.

This file connects physical column ranges to the values selected by the
witness layout. It does not prove row satisfaction or protocol balance.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessAccess

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.BitWitness
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout
open Nightstream.Protocol.Nebula.Fingerprint

private theorem segmentOffset_exact : XOffset.segment = 0 := rfl
private theorem stepOffset_exact : XOffset.step = 16 := rfl
private theorem timestampInOffset_exact : XOffset.timestampIn = 32 := rfl
private theorem timestampOutOffset_exact : XOffset.timestampOut = 76 := rfl
private theorem gammaOffset_exact : XOffset.gamma = 120 := rfl
private theorem productsInOffset_exact : XOffset.productsIn = 376 := rfl
private theorem productsOutOffset_exact : XOffset.productsOut = 888 := rfl
private theorem segmentBits_exact : segmentIndexBits = 16 := rfl
private theorem stepBits_exact : stepIndexBits = 16 := rfl
private theorem timestampBits_exact :
    Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits = 44 := rfl
private theorem limbBits_exact : extensionLimbBits = 64 := rfl
private theorem totalExtensionBits_exact : extensionBits = 128 := rfl
private theorem addressBits_exact : wasm42x6.addressBits = 10 := rfl
private theorem countBits_exact : wasm42x6.countBits = 1 := rfl
private theorem operationBits_exact : wasm42x6.operationBits = 121 := rfl
private theorem operationAuxiliaryBits_exact :
    wasm42x6.operationAuxiliaryBits = 301 := rfl
private theorem scanAuxiliaryBits_exact : scanAuxiliaryBits = 256 := rfl

theorem assignment_public
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < publicInputBits) :
    assignment challengeValues batch (xColumn offset) =
      publicBit challengeValues batch offset := by
  change offset < 1400 at offsetBound
  unfold assignment xColumn
  simp only [wasm42x6_publicColumns]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  simpa using congrArg (publicBit challengeValues batch)
    (show 1 + offset - 1 = offset by omega)

theorem assignment_operation
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < wasm42x6.operationBits) :
    assignment challengeValues batch (wasm42x6.operationLane + offset) =
      operationBit batch offset := by
  change offset < 121 at offsetBound
  unfold assignment
  simp only [wasm42x6_operationLane, wasm42x6_publicColumns,
    wasm42x6_operationBits]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  simpa using congrArg (operationBit batch)
    (show 1404 + offset - 1404 = offset by omega)

theorem assignment_initial_cell
    (challengeValues : Challenges) (batch : Fin 2)
    (slot offset : Nat) (slotBound : slot < wasm42x6.scanSlots)
    (offsetBound : offset < cellBits) :
    assignment challengeValues batch
        (wasm42x6.initialScanLane + slot * cellBits + offset) =
      cellBit (cellAt batch false slot) offset := by
  change slot < 1024 at slotBound
  change offset < 76 at offsetBound
  unfold assignment
  simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
    wasm42x6_operationBits, wasm42x6_initialScanLane,
    wasm42x6_scanSlots, cellBits_exact]
  have rangeBound : slot * 76 + offset < 1024 * 76 := by
    omega
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  rw [show 1566 + slot * 76 + offset - 1566 =
      slot * 76 + offset by omega]
  exact cellVectorBit_at (cellAt batch false) slot offset offsetBound

theorem assignment_final_cell
    (challengeValues : Challenges) (batch : Fin 2)
    (slot offset : Nat) (slotBound : slot < wasm42x6.scanSlots)
    (offsetBound : offset < cellBits) :
    assignment challengeValues batch
        (wasm42x6.finalScanLane + slot * cellBits + offset) =
      cellBit (cellAt batch true slot) offset := by
  change slot < 1024 at slotBound
  change offset < 76 at offsetBound
  unfold assignment
  simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
    wasm42x6_operationBits, wasm42x6_initialScanLane,
    wasm42x6_finalScanLane, wasm42x6_scanSlots, cellBits_exact]
  have rangeBound : slot * 76 + offset < 1024 * 76 := by
    omega
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  rw [show 79434 + slot * 76 + offset - 79434 =
      slot * 76 + offset by omega]
  exact cellVectorBit_at (cellAt batch true) slot offset offsetBound

theorem assignment_operation_auxiliary
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < wasm42x6.operationAuxiliaryBits) :
    assignment challengeValues batch (wasm42x6.auxiliaryStart + offset) =
      operationAuxiliaryBit challengeValues batch offset := by
  change offset < 301 at offsetBound
  unfold assignment
  simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
    wasm42x6_operationBits, wasm42x6_initialScanLane,
    wasm42x6_finalScanLane, wasm42x6_auxiliaryStart,
    wasm42x6_scanAuxiliaryStart, wasm42x6_scanSlots, cellBits_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  simpa using congrArg (operationAuxiliaryBit challengeValues batch)
    (show 157302 + offset - 157302 = offset by omega)

theorem assignment_scan_auxiliary
    (challengeValues : Challenges) (batch : Fin 2)
    (offset : Nat) (offsetBound : offset < wasm42x6.scanSlots *
      scanAuxiliaryBits) :
    assignment challengeValues batch
        (wasm42x6.scanAuxiliaryStart + offset) =
      scanAuxiliaryBit challengeValues batch offset := by
  change offset < 262144 at offsetBound
  unfold assignment
  simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
    wasm42x6_operationBits, wasm42x6_initialScanLane,
    wasm42x6_finalScanLane, wasm42x6_auxiliaryStart,
    wasm42x6_scanAuxiliaryStart, wasm42x6_columnCount,
    wasm42x6_scanSlots, cellBits_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  simpa using congrArg (scanAuxiliaryBit challengeValues batch)
    (show 157603 + offset - 157603 = offset by omega)

/-- Every alignment column declared by the selected physical layout carries
zero in the honest assignment. -/
theorem filler_get
    (challengeValues : Challenges) (batch : Fin 2)
    (column : Nat) (member : column ∈ wasm42x6.fillerColumns) :
    assignment challengeValues batch column = 0 := by
  unfold Params.fillerColumns at member
  rw [List.mem_append] at member
  rcases member with firstThree | fourth
  · rw [List.mem_append] at firstThree
    rcases firstThree with firstTwo | third
    · rw [List.mem_append] at firstTwo
      rcases firstTwo with first | second
      · rcases List.mem_range'.mp first with
          ⟨offset, offsetBound, columnExact⟩
        simp only [wasm42x6_publicColumns, wasm42x6_operationLane]
          at offsetBound columnExact
        rw [columnExact]
        unfold assignment
        simp only [wasm42x6_publicColumns, wasm42x6_operationLane]
        rw [if_neg (by omega), if_neg (by omega), if_pos (by omega)]
      · rcases List.mem_range'.mp second with
          ⟨offset, offsetBound, columnExact⟩
        simp only [wasm42x6_operationLane, wasm42x6_initialScanLane,
          wasm42x6_operationBits, wasm42x6_operationSlots]
          at offsetBound columnExact
        rw [columnExact]
        unfold assignment
        simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
          wasm42x6_operationBits, wasm42x6_initialScanLane]
        rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
          if_neg (by omega), if_pos (by omega)]
    · rcases List.mem_range'.mp third with
        ⟨offset, offsetBound, columnExact⟩
      simp only [wasm42x6_initialScanLane, wasm42x6_finalScanLane,
        wasm42x6_scanSlots, cellBits_exact] at offsetBound columnExact
      rw [columnExact]
      unfold assignment
      simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
        wasm42x6_operationBits, wasm42x6_initialScanLane,
        wasm42x6_finalScanLane, wasm42x6_scanSlots, cellBits_exact]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
        if_neg (by omega), if_neg (by omega), if_neg (by omega),
        if_pos (by omega)]
  · rcases List.mem_range'.mp fourth with
      ⟨offset, offsetBound, columnExact⟩
    simp only [wasm42x6_finalScanLane, wasm42x6_auxiliaryStart,
      wasm42x6_scanSlots, cellBits_exact] at offsetBound columnExact
    rw [columnExact]
    unfold assignment
    simp only [wasm42x6_publicColumns, wasm42x6_operationLane,
      wasm42x6_operationBits, wasm42x6_initialScanLane,
      wasm42x6_finalScanLane, wasm42x6_auxiliaryStart,
      wasm42x6_scanSlots, cellBits_exact]
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
      if_neg (by omega), if_neg (by omega), if_neg (by omega),
      if_neg (by omega), if_neg (by omega), if_pos (by omega)]

/-! ## Public words -/

theorem segment_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < segmentIndexBits) :
    assignment challengeValues batch
        (xColumn (XOffset.segment + bit)) = bitField 0 bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact, Nat.zero_add]
  rw [if_pos (by omega)]

theorem step_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < stepIndexBits) :
    assignment challengeValues batch
        (xColumn (XOffset.step + bit)) = bitField batch.val bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact, timestampInOffset_exact]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  congr 1
  omega

theorem timestamp_in_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit <
      Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) :
    assignment challengeValues batch
        (xColumn (XOffset.timestampIn + bit)) = bitField batch.val bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  congr 1
  omega

theorem timestamp_out_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit <
      Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) :
    assignment challengeValues batch
        (xColumn (XOffset.timestampOut + bit)) = bitField 1 bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  congr 1
  omega

theorem gamma_get
    (challengeValues : Challenges) (batch : Fin 2)
    (challenge component bit : Nat)
    (challengeBound : challenge < 2) (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (xColumn (XOffset.gamma + challenge * extensionBits +
          component * extensionLimbBits + bit)) =
      bitField
        (kComponentValue (challengeAt challengeValues challenge) component)
        bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact, totalExtensionBits_exact,
      limbBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  rw [show 120 + challenge * 128 + component * 64 + bit - 120 =
      challenge * 128 + component * 64 + bit by omega]
  exact kVectorBit_at (challengeAt challengeValues) challenge component bit
    componentBound bitBound

theorem product_input_get
    (challengeValues : Challenges) (batch : Fin 2)
    (product component bit : Nat)
    (productBound : product < 4) (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (xColumn (XOffset.productsIn + product * extensionBits +
          component * extensionLimbBits + bit)) =
      bitField
        (kComponentValue
          (inputProductAt challengeValues batch product) component) bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact, totalExtensionBits_exact,
      limbBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by omega)]
  rw [show 376 + product * 128 + component * 64 + bit - 376 =
      product * 128 + component * 64 + bit by omega]
  exact kVectorBit_at (inputProductAt challengeValues batch) product
    component bit componentBound bitBound

theorem product_output_get
    (challengeValues : Challenges) (batch : Fin 2)
    (product component bit : Nat)
    (productBound : product < 4) (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (xColumn (XOffset.productsOut + product * extensionBits +
          component * extensionLimbBits + bit)) =
      bitField
        (kComponentValue
          (outputProductAt challengeValues batch product) component) bit := by
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact, publicInputBits_exact,
    segmentBits_exact, stepBits_exact, timestampBits_exact, limbBits_exact,
    totalExtensionBits_exact] at *
  rw [assignment_public challengeValues batch _ (by
    simp only [publicInputBits_exact, totalExtensionBits_exact,
      limbBits_exact]
    omega)]
  unfold publicBit
  simp only [segmentOffset_exact, stepOffset_exact,
    timestampInOffset_exact, timestampOutOffset_exact, gammaOffset_exact,
    productsInOffset_exact, productsOutOffset_exact]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by omega)]
  rw [show 888 + product * 128 + component * 64 + bit - 888 =
      product * 128 + component * 64 + bit by omega]
  exact kVectorBit_at (outputProductAt challengeValues batch) product
    component bit componentBound bitBound

/-! ## Operation words -/

theorem operation_pad_get
    (challengeValues : Challenges) (batch : Fin 2) :
    assignment challengeValues batch (wasm42x6.operationSlot 0) =
      bitField (if isSecond batch then 1 else 0) 0 := by
  rw [show wasm42x6.operationSlot 0 = wasm42x6.operationLane + 0 by rfl,
    assignment_operation challengeValues batch 0 (by
      simp only [operationBits_exact]
      omega)]
  simp [operationBit]

theorem operation_is_write_get
    (challengeValues : Challenges) (batch : Fin 2) :
    assignment challengeValues batch (wasm42x6.operationSlot 0 + 1) =
      bitField 0 0 := by
  rw [show wasm42x6.operationSlot 0 + 1 =
      wasm42x6.operationLane + 1 by rfl,
    assignment_operation challengeValues batch 1 (by
      simp only [operationBits_exact]
      omega)]
  simp [operationBit]

theorem operation_ram_get
    (challengeValues : Challenges) (batch : Fin 2) :
    assignment challengeValues batch (wasm42x6.operationSlot 0 + 2) =
      bitField (if isSecond batch then 0 else 1) 0 := by
  rw [show wasm42x6.operationSlot 0 + 2 =
      wasm42x6.operationLane + 2 by rfl,
    assignment_operation challengeValues batch 2 (by
      simp only [operationBits_exact]
      omega)]
  simp [operationBit]

theorem operation_address_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < wasm42x6.addressBits) :
    assignment challengeValues batch
        (wasm42x6.operationSlot 0 + 3 + bit) = bitField 0 bit := by
  simp only [addressBits_exact] at bitBound
  rw [show wasm42x6.operationSlot 0 + 3 + bit =
      wasm42x6.operationLane + (3 + bit) by
        unfold Params.operationSlot
        omega,
    assignment_operation challengeValues batch (3 + bit) (by
      simp only [operationBits_exact]
      omega)]
  unfold operationBit
  simp only [addressBits_exact]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_pos (by omega)]
  congr 1
  omega

theorem operation_read_value_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < valueBits) :
    assignment challengeValues batch
        (wasm42x6.operationSlot 0 + 3 + wasm42x6.addressBits + bit) =
      bitField (if isSecond batch then 0 else 42) bit := by
  simp only [addressBits_exact, valueBits] at bitBound ⊢
  rw [show wasm42x6.operationSlot 0 + 3 + 10 + bit =
      wasm42x6.operationLane + (13 + bit) by
        unfold Params.operationSlot
        omega,
    assignment_operation challengeValues batch (13 + bit) (by
      simp only [operationBits_exact]
      omega)]
  unfold operationBit
  simp only [addressBits_exact, valueBits]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_neg (by omega), if_pos (by omega)]
  congr 1
  omega

theorem operation_write_value_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < valueBits) :
    assignment challengeValues batch
        (wasm42x6.operationSlot 0 + 3 + wasm42x6.addressBits + valueBits +
          bit) =
      bitField (if isSecond batch then 0 else 42) bit := by
  simp only [addressBits_exact, valueBits] at bitBound ⊢
  rw [show wasm42x6.operationSlot 0 + 3 + 10 + 32 + bit =
      wasm42x6.operationLane + (45 + bit) by
        unfold Params.operationSlot
        omega,
    assignment_operation challengeValues batch (45 + bit) (by
      simp only [operationBits_exact]
      omega)]
  unfold operationBit
  simp only [addressBits_exact, valueBits]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_neg (by omega), if_neg (by omega), if_pos (by omega)]
  congr 1
  omega

theorem operation_read_timestamp_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit <
      Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) :
    assignment challengeValues batch
        (wasm42x6.operationSlot 0 + 3 + wasm42x6.addressBits +
          2 * valueBits + bit) = bitField 0 bit := by
  simp only [addressBits_exact, valueBits, timestampBits_exact] at bitBound ⊢
  rw [show wasm42x6.operationSlot 0 + 3 + 10 + 2 * 32 + bit =
      wasm42x6.operationLane + (77 + bit) by
        unfold Params.operationSlot
        omega,
    assignment_operation challengeValues batch (77 + bit) (by
      simp only [operationBits_exact]
      omega)]
  unfold operationBit
  simp only [addressBits_exact, valueBits]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega),
    if_neg (by omega), if_neg (by omega), if_neg (by omega)]
  congr 1
  omega

/-! ## Operation auxiliaries -/

theorem operation_diff_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit <
      Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) :
    assignment challengeValues batch
        (wasm42x6.operationDiff 0 + bit) = bitField 0 bit := by
  simp only [timestampBits_exact] at bitBound
  rw [show wasm42x6.operationDiff 0 + bit =
      wasm42x6.auxiliaryStart + bit by rfl,
    assignment_operation_auxiliary challengeValues batch bit (by
      simp only [operationAuxiliaryBits_exact]
      omega)]
  unfold operationAuxiliaryBit
  simp only [timestampBits_exact]
  rw [if_pos (by omega)]

theorem operation_count_get
    (challengeValues : Challenges) (batch : Fin 2)
    (bit : Nat) (bitBound : bit < wasm42x6.countBits) :
    assignment challengeValues batch
        (wasm42x6.operationCount 0 + bit) =
      bitField (if isSecond batch then 0 else 1) bit := by
  simp only [countBits_exact] at bitBound
  rw [show wasm42x6.operationCount 0 + bit =
      wasm42x6.auxiliaryStart + (44 + bit) by
        unfold Params.operationCount Params.operationDiff
          Params.operationAuxiliary
        simp only [timestampBits_exact]
        omega,
    assignment_operation_auxiliary challengeValues batch (44 + bit) (by
      simp only [operationAuxiliaryBits_exact]
      omega)]
  unfold operationAuxiliaryBit
  simp only [timestampBits_exact, countBits_exact]
  rw [if_neg (by omega), if_pos (by omega)]
  congr 1
  omega

theorem operation_read_product_get
    (challengeValues : Challenges) (batch : Fin 2)
    (component bit : Nat) (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (wasm42x6.operationReadProduct 0 +
          component * extensionLimbBits + bit) =
      bitField
        (kComponentValue
          (operationProductAt challengeValues batch false) component) bit := by
  simp only [limbBits_exact] at bitBound ⊢
  rw [show wasm42x6.operationReadProduct 0 + component * 64 + bit =
      wasm42x6.auxiliaryStart + (45 + component * 64 + bit) by
        unfold Params.operationReadProduct Params.operationCount
          Params.operationDiff Params.operationAuxiliary
        simp only [timestampBits_exact, countBits_exact]
        omega,
    assignment_operation_auxiliary challengeValues batch
      (45 + component * 64 + bit) (by
        simp only [operationAuxiliaryBits_exact]
        omega)]
  unfold operationAuxiliaryBit
  simp only [timestampBits_exact, countBits_exact,
    totalExtensionBits_exact]
  rw [if_neg (by omega), if_neg (by omega), if_pos (by omega)]
  rw [show 45 + component * 64 + bit - 45 =
      component * 64 + bit by omega]
  exact kBit_at _ component bit componentBound bitBound

theorem operation_write_product_get
    (challengeValues : Challenges) (batch : Fin 2)
    (component bit : Nat) (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (wasm42x6.operationWriteProduct 0 +
          component * extensionLimbBits + bit) =
      bitField
        (kComponentValue
          (operationProductAt challengeValues batch true) component) bit := by
  simp only [limbBits_exact] at bitBound ⊢
  rw [show wasm42x6.operationWriteProduct 0 + component * 64 + bit =
      wasm42x6.auxiliaryStart + (173 + component * 64 + bit) by
        unfold Params.operationWriteProduct Params.operationReadProduct
          Params.operationCount Params.operationDiff Params.operationAuxiliary
        simp only [timestampBits_exact, countBits_exact,
          totalExtensionBits_exact]
        omega,
    assignment_operation_auxiliary challengeValues batch
      (173 + component * 64 + bit) (by
        simp only [operationAuxiliaryBits_exact]
        omega)]
  unfold operationAuxiliaryBit
  simp only [timestampBits_exact, countBits_exact,
    totalExtensionBits_exact]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega)]
  rw [show 173 + component * 64 + bit - 173 =
      component * 64 + bit by omega]
  exact kBit_at _ component bit componentBound bitBound

/-! ## Scan words -/

theorem scan_value_get
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (slot bit : Nat) (slotBound : slot < 1024)
    (bitBound : bit < valueBits) :
    assignment challengeValues batch
        (scanCellStart wasm42x6 final slot + bit) =
      bitField (cellAt batch final slot).value bit := by
  simp only [valueBits] at bitBound
  cases final
  · rw [show scanCellStart wasm42x6 false slot + bit =
      wasm42x6.initialScanLane + slot * cellBits + bit by
        unfold scanCellStart
        rfl,
      assignment_initial_cell challengeValues batch slot bit slotBound (by
        simp only [cellBits_exact]
        omega)]
    unfold cellBit
    simp only [valueBits]
    rw [if_pos (by omega)]
  · rw [show scanCellStart wasm42x6 true slot + bit =
      wasm42x6.finalScanLane + slot * cellBits + bit by
        unfold scanCellStart
        rfl,
      assignment_final_cell challengeValues batch slot bit slotBound (by
        simp only [cellBits_exact]
        omega)]
    unfold cellBit
    simp only [valueBits]
    rw [if_pos (by omega)]

theorem scan_timestamp_get
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (slot bit : Nat) (slotBound : slot < 1024)
    (bitBound : bit <
      Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) :
    assignment challengeValues batch
        (scanCellStart wasm42x6 final slot + valueBits + bit) =
      bitField (cellAt batch final slot).timestamp bit := by
  simp only [valueBits, timestampBits_exact] at bitBound ⊢
  cases final
  · rw [show scanCellStart wasm42x6 false slot + 32 + bit =
      wasm42x6.initialScanLane + slot * cellBits + (32 + bit) by
        simp [scanCellStart]
        omega,
      assignment_initial_cell challengeValues batch slot (32 + bit)
        slotBound (by simp only [cellBits_exact]; omega)]
    unfold cellBit
    simp only [valueBits]
    rw [if_neg (by omega)]
    congr 1
    omega
  · rw [show scanCellStart wasm42x6 true slot + 32 + bit =
      wasm42x6.finalScanLane + slot * cellBits + (32 + bit) by
        simp [scanCellStart]
        omega,
      assignment_final_cell challengeValues batch slot (32 + bit)
        slotBound (by simp only [cellBits_exact]; omega)]
    unfold cellBit
    simp only [valueBits]
    rw [if_neg (by omega)]
    congr 1
    omega

private theorem scan_auxiliary_div
    (slot localOffset : Nat) (localBound : localOffset < 256) :
    (slot * 256 + localOffset) / 256 = slot := by
  rw [Nat.mul_comm slot 256, Nat.mul_add_div (by decide),
    Nat.div_eq_of_lt localBound]
  omega

private theorem scan_auxiliary_mod
    (slot localOffset : Nat) (localBound : localOffset < 256) :
    (slot * 256 + localOffset) % 256 = localOffset :=
  Nat.mul_add_mod_of_lt localBound

theorem initial_scan_product_get
    (challengeValues : Challenges) (batch : Fin 2)
    (slot component bit : Nat) (slotBound : slot < 1024)
    (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (wasm42x6.initialScanProduct slot +
          component * extensionLimbBits + bit) =
      bitField
        (kComponentValue
          (scanProductAt challengeValues batch false (slot + 1)) component)
        bit := by
  simp only [limbBits_exact] at bitBound ⊢
  rw [show wasm42x6.initialScanProduct slot + component * 64 + bit =
      wasm42x6.scanAuxiliaryStart +
        (slot * 256 + (component * 64 + bit)) by
          unfold Params.initialScanProduct
          simp only [scanAuxiliaryBits_exact]
          omega,
    assignment_scan_auxiliary challengeValues batch
      (slot * 256 + (component * 64 + bit)) (by
        simp only [wasm42x6_scanSlots, scanAuxiliaryBits_exact]
        omega)]
  have localBound : component * 64 + bit < 256 := by omega
  unfold scanAuxiliaryBit
  simp only [scanAuxiliaryBits_exact, totalExtensionBits_exact,
    scan_auxiliary_div slot _ localBound,
    scan_auxiliary_mod slot _ localBound]
  rw [if_pos (by omega)]
  exact kBit_at _ component bit componentBound bitBound

theorem final_scan_product_get
    (challengeValues : Challenges) (batch : Fin 2)
    (slot component bit : Nat) (slotBound : slot < 1024)
    (componentBound : component < 2)
    (bitBound : bit < extensionLimbBits) :
    assignment challengeValues batch
        (wasm42x6.finalScanProduct slot +
          component * extensionLimbBits + bit) =
      bitField
        (kComponentValue
          (scanProductAt challengeValues batch true (slot + 1)) component)
        bit := by
  simp only [limbBits_exact] at bitBound ⊢
  rw [show wasm42x6.finalScanProduct slot + component * 64 + bit =
      wasm42x6.scanAuxiliaryStart +
        (slot * 256 + (128 + component * 64 + bit)) by
          unfold Params.finalScanProduct Params.initialScanProduct
          simp only [scanAuxiliaryBits_exact, totalExtensionBits_exact]
          omega,
    assignment_scan_auxiliary challengeValues batch
      (slot * 256 + (128 + component * 64 + bit)) (by
        simp only [wasm42x6_scanSlots, scanAuxiliaryBits_exact]
        omega)]
  have localBound : 128 + component * 64 + bit < 256 := by omega
  unfold scanAuxiliaryBit
  simp only [scanAuxiliaryBits_exact, totalExtensionBits_exact,
    scan_auxiliary_div slot _ localBound,
    scan_auxiliary_mod slot _ localBound]
  rw [if_neg (by omega)]
  rw [show 128 + component * 64 + bit - 128 =
      component * 64 + bit by omega]
  exact kBit_at _ component bit componentBound bitBound

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessAccess
