import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessAccess

/-!
Exact word values carried by the modular 42-times-6 Nebula witness.

Assurance tier: model-level.

This file converts the physical bit-range access theorems into exact base-
and extension-field word values. It does not prove row satisfaction,
terminal balance, F-prime placement, Rust conformance, or a security result.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessWords

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.Lowering.Nebula.BitWitness
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessAccess
open Nightstream.Protocol.Nebula.Fingerprint

private abbrev Lin := Rows.LinearCombination

private theorem fin_lt_two_pow_limb (value : F) :
    value.val < 2 ^ extensionLimbBits := by
  have modulusBound : goldilocksModulus < 2 ^ extensionLimbBits := by
    decide
  exact Nat.lt_trans value.isLt modulusBound

private theorem kComponentValue_lt_two_pow_limb
    (value : K) (component : Nat) :
    kComponentValue value component < 2 ^ extensionLimbBits := by
  cases component <;> exact fin_lt_two_pow_limb _

private theorem eval_limb_exact
    (assignment : Nat -> F) (start : Nat) (value : F)
    (get : forall bit, bit < extensionLimbBits ->
      assignment (start + bit) = bitField value.val bit) :
    Rows.LinearCombination.eval assignment
        (Rows.LinearCombination.word start extensionLimbBits) = value := by
  calc
    Rows.LinearCombination.eval assignment
        (Rows.LinearCombination.word start extensionLimbBits) =
        fieldOfNat value.val :=
      eval_word_exact_of_get assignment start extensionLimbBits value.val get
        (fin_lt_two_pow_limb value)
    _ = value := fieldOfNat_finVal value

private theorem eval_nat_exact
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall bit, bit < width ->
      assignment (start + bit) = bitField value bit)
    (bound : value < 2 ^ width) :
    Rows.LinearCombination.eval assignment
        (Rows.LinearCombination.word start width) = fieldOfNat value :=
  eval_word_exact_of_get assignment start width value get bound

/-! ## Public words -/

theorem step_word_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (publicWord XOffset.step stepIndexBits) = fieldOfNat batch.val := by
  apply eval_nat_exact
  · intro bit bitBound
    rw [show xColumn XOffset.step + bit =
        xColumn (XOffset.step + bit) by
      unfold xColumn
      omega]
    exact step_get challengeValues batch bit bitBound
  · have := batch.isLt
    simp only [stepIndexBits]
    omega

theorem timestamp_in_word_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch) timestampIn =
      fieldOfNat batch.val := by
  apply eval_nat_exact
  · intro bit bitBound
    rw [show xColumn XOffset.timestampIn + bit =
        xColumn (XOffset.timestampIn + bit) by
      unfold xColumn
      omega]
    exact timestamp_in_get challengeValues batch bit bitBound
  · have := batch.isLt
    simp only [Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits]
    omega

theorem timestamp_out_word_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (publicWord XOffset.timestampOut
          Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits) =
      fieldOfNat 1 := by
  apply eval_nat_exact
  · intro bit bitBound
    rw [show xColumn XOffset.timestampOut + bit =
        xColumn (XOffset.timestampOut + bit) by
      unfold xColumn
      omega]
    exact timestamp_out_get challengeValues batch bit bitBound
  · decide

theorem gamma_component_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (challenge component : Nat) (challengeBound : challenge < 2)
    (componentBound : component < 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (gammaWord challenge component) =
      fieldOfNat
        (kComponentValue (challengeAt challengeValues challenge) component) := by
  apply eval_nat_exact
  · intro bit bitBound
    rw [show xColumn
          (XOffset.gamma + challenge * extensionBits +
            component * extensionLimbBits) + bit =
        xColumn
          (XOffset.gamma + challenge * extensionBits +
            component * extensionLimbBits + bit) by
      unfold xColumn
      omega]
    exact gamma_get challengeValues batch challenge component bit
      challengeBound componentBound bitBound
  · exact kComponentValue_lt_two_pow_limb _ _

private theorem challenges_ext
    {left right : Challenges}
    (gamma1 : left.gamma1 = right.gamma1)
    (gamma2 : left.gamma2 = right.gamma2) :
    left = right := by
  cases left
  cases right
  simp_all

theorem challenges_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    challenges (assignment challengeValues batch) = challengeValues := by
  apply challenges_ext
  · change evaluatePair _ _ _ = challengeValues.gamma1
    change K.mk _ _ = K.mk challengeValues.gamma1.c0 challengeValues.gamma1.c1
    apply congrArg₂ K.mk
    · have low := gamma_component_exact challengeValues batch 0 0
          (by decide) (by decide)
      simp only [challengeAt, kComponentValue] at low
      rw [fieldOfNat_finVal] at low
      exact low
    · have high := gamma_component_exact challengeValues batch 0 1
          (by decide) (by decide)
      simp only [challengeAt, kComponentValue] at high
      rw [fieldOfNat_finVal] at high
      exact high
  · change evaluatePair _ _ _ = challengeValues.gamma2
    change K.mk _ _ = K.mk challengeValues.gamma2.c0 challengeValues.gamma2.c1
    apply congrArg₂ K.mk
    · have low := gamma_component_exact challengeValues batch 1 0
          (by decide) (by decide)
      simp only [challengeAt, kComponentValue] at low
      rw [fieldOfNat_finVal] at low
      exact low
    · have high := gamma_component_exact challengeValues batch 1 1
          (by decide) (by decide)
      simp only [challengeAt, kComponentValue] at high
      rw [fieldOfNat_finVal] at high
      exact high

private theorem public_product_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (product : Nat) (productBound : product < 4) (output : Bool) :
    evaluatePair (assignment challengeValues batch)
        (if output then productOutputWord product 0
          else productInputWord product 0)
        (if output then productOutputWord product 1
          else productInputWord product 1) =
      if output then outputProductAt challengeValues batch product
      else inputProductAt challengeValues batch product := by
  let value := if output then outputProductAt challengeValues batch product
    else inputProductAt challengeValues batch product
  have component (component : Nat) (componentBound : component < 2) :
      Rows.LinearCombination.eval (assignment challengeValues batch)
          (if output then productOutputWord product component
            else productInputWord product component) =
        fieldOfNat (kComponentValue value component) := by
    cases output
    · apply eval_nat_exact
      · intro bit bitBound
        rw [show xColumn
              (XOffset.productsIn + product * extensionBits +
                component * extensionLimbBits) + bit =
            xColumn
              (XOffset.productsIn + product * extensionBits +
                component * extensionLimbBits + bit) by
          unfold xColumn
          omega]
        simpa [value] using
          product_input_get challengeValues batch product component bit
            productBound componentBound bitBound
      · simpa [value] using kComponentValue_lt_two_pow_limb
          (inputProductAt challengeValues batch product) component
    · apply eval_nat_exact
      · intro bit bitBound
        rw [show xColumn
              (XOffset.productsOut + product * extensionBits +
                component * extensionLimbBits) + bit =
            xColumn
              (XOffset.productsOut + product * extensionBits +
                component * extensionLimbBits + bit) by
          unfold xColumn
          omega]
        simpa [value] using
          product_output_get challengeValues batch product component bit
            productBound componentBound bitBound
      · simpa [value] using kComponentValue_lt_two_pow_limb
          (outputProductAt challengeValues batch product) component
  cases output
  · change K.mk _ _ = K.mk
      (inputProductAt challengeValues batch product).c0
      (inputProductAt challengeValues batch product).c1
    apply congrArg₂ K.mk
    · have low := component 0 (by decide)
      simp only [value, kComponentValue] at low
      rw [fieldOfNat_finVal] at low
      exact low
    · have high := component 1 (by decide)
      simp only [value, kComponentValue] at high
      rw [fieldOfNat_finVal] at high
      exact high
  · change K.mk _ _ = K.mk
      (outputProductAt challengeValues batch product).c0
      (outputProductAt challengeValues batch product).c1
    apply congrArg₂ K.mk
    · have low := component 0 (by decide)
      simp only [value, kComponentValue] at low
      rw [fieldOfNat_finVal] at low
      exact low
    · have high := component 1 (by decide)
      simp only [value, kComponentValue] at high
      rw [fieldOfNat_finVal] at high
      exact high

theorem input_product_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (product : Nat) (productBound : product < 4) :
    inputProduct (assignment challengeValues batch) product =
      inputProductAt challengeValues batch product := by
  exact public_product_exact challengeValues batch product productBound false

theorem output_product_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (product : Nat) (productBound : product < 4) :
    outputProduct (assignment challengeValues batch) product =
      outputProductAt challengeValues batch product := by
  exact public_product_exact challengeValues batch product productBound true

/-! ## Operation words -/

theorem operation_scalar_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (field : Nat) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (match field with
        | 0 => operationPad wasm42x6 0
        | 1 => operationIsWrite wasm42x6 0
        | _ => operationRam wasm42x6 0) =
      fieldOfNat
        (match field with
        | 0 => if isSecond batch then 1 else 0
        | 1 => 0
        | _ => if isSecond batch then 0 else 1) := by
  cases field with
  | zero =>
      cases second : isSecond batch <;>
        simpa [second, operationPad, Rows.LinearCombination.eval,
          Rows.LinearCombination.bit] using
          operation_pad_get challengeValues batch
  | succ field =>
      cases field with
      | zero =>
          simpa [operationIsWrite, Rows.LinearCombination.eval,
            Rows.LinearCombination.bit] using
            operation_is_write_get challengeValues batch
      | succ field =>
          cases second : isSecond batch <;>
            simpa [second, operationRam, Rows.LinearCombination.eval,
              Rows.LinearCombination.bit] using
              operation_ram_get challengeValues batch

theorem operation_address_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationAddress wasm42x6 0) = fieldOfNat 0 := by
  apply eval_nat_exact
  · exact operation_address_get challengeValues batch
  · decide

theorem operation_read_value_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationReadValue wasm42x6 0) =
      fieldOfNat (if isSecond batch then 0 else 42) := by
  apply eval_nat_exact
  · exact operation_read_value_get challengeValues batch
  · simp only [valueBits]
    split <;> decide

theorem operation_write_value_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationWriteValue wasm42x6 0) =
      fieldOfNat (if isSecond batch then 0 else 42) := by
  apply eval_nat_exact
  · exact operation_write_value_get challengeValues batch
  · simp only [valueBits]
    split <;> decide

theorem operation_read_timestamp_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationReadTimestamp wasm42x6 0) = fieldOfNat 0 := by
  apply eval_nat_exact
  · exact operation_read_timestamp_get challengeValues batch
  · decide

theorem operation_diff_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationDiffWord wasm42x6 0) = fieldOfNat 0 := by
  apply eval_nat_exact
  · exact operation_diff_get challengeValues batch
  · decide

theorem operation_count_exact
    (challengeValues : Challenges) (batch : Fin 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (operationCountWord wasm42x6 0) =
      fieldOfNat (if isSecond batch then 0 else 1) := by
  apply eval_nat_exact
  · exact operation_count_get challengeValues batch
  · simp only [wasm42x6_countBits]
    split <;> decide

private theorem operation_product_component_exact
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool)
    (component : Nat) (componentBound : component < 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (if write then operationWriteProductWord wasm42x6 0 component
          else operationReadProductWord wasm42x6 0 component) =
      fieldOfNat (kComponentValue
        (operationProductAt challengeValues batch write) component) := by
  cases write
  · apply eval_nat_exact
    · intro bit bitBound
      exact operation_read_product_get challengeValues batch component bit
        componentBound bitBound
    · exact kComponentValue_lt_two_pow_limb _ _
  · apply eval_nat_exact
    · intro bit bitBound
      exact operation_write_product_get challengeValues batch component bit
        componentBound bitBound
    · exact kComponentValue_lt_two_pow_limb _ _

theorem operation_product_exact
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool) :
    operationProduct (assignment challengeValues batch) wasm42x6 0 write =
      operationProductAt challengeValues batch write := by
  change K.mk _ _ = K.mk
    (operationProductAt challengeValues batch write).c0
    (operationProductAt challengeValues batch write).c1
  apply congrArg₂ K.mk
  · have low := operation_product_component_exact challengeValues batch
        write 0 (by decide)
    simp only [kComponentValue] at low
    rw [fieldOfNat_finVal] at low
    exact low
  · have high := operation_product_component_exact challengeValues batch
        write 1 (by decide)
    simp only [kComponentValue] at high
    rw [fieldOfNat_finVal] at high
    exact high

/-! ## Scan words -/

theorem scan_value_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (slot : Nat) (slotBound : slot < 1024) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (scanValue wasm42x6 final slot) =
      fieldOfNat (cellAt batch final slot).value := by
  apply eval_nat_exact
  · intro bit bitBound
    exact scan_value_get challengeValues batch final slot bit slotBound bitBound
  · have valueBound : (cellAt batch final slot).value <= 42 := by
      cases final <;>
        by_cases second : isSecond batch = true <;>
        by_cases first : slot = 0 <;>
        simp [cellAt, firstCell, secondCell, second, Memory.initialRamAt,
          Memory.finalRamAt, Memory.readCell, Memory.writeCell,
          Memory.blankCell, first]
    simp only [valueBits]
    omega

theorem scan_timestamp_exact
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (slot : Nat) (slotBound : slot < 1024) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (scanTimestamp wasm42x6 final slot) =
      fieldOfNat (cellAt batch final slot).timestamp := by
  apply eval_nat_exact
  · intro bit bitBound
    exact scan_timestamp_get challengeValues batch final slot bit slotBound
      bitBound
  · have timestampBound : (cellAt batch final slot).timestamp <= 1 := by
      cases final <;>
        by_cases second : isSecond batch = true <;>
        by_cases first : slot = 0 <;>
        simp [cellAt, firstCell, secondCell, second, Memory.initialRamAt,
          Memory.finalRamAt, Memory.readCell, Memory.writeCell,
          Memory.blankCell, first]
    simp only [Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits]
    omega

private theorem scan_product_component_exact
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot component : Nat) (slotBound : slot < 1024)
    (componentBound : component < 2) :
    Rows.LinearCombination.eval (assignment challengeValues batch)
        (scanProductWord wasm42x6 final slot component) =
      fieldOfNat (kComponentValue
        (scanProductAt challengeValues batch final (slot + 1)) component) := by
  cases final
  · apply eval_nat_exact
    · intro bit bitBound
      exact initial_scan_product_get challengeValues batch slot component bit
        slotBound componentBound bitBound
    · exact kComponentValue_lt_two_pow_limb _ _
  · apply eval_nat_exact
    · intro bit bitBound
      exact final_scan_product_get challengeValues batch slot component bit
        slotBound componentBound bitBound
    · exact kComponentValue_lt_two_pow_limb _ _

theorem scan_product_exact
    (challengeValues : Challenges) (batch : Fin 2) (final : Bool)
    (slot : Nat) (slotBound : slot < 1024) :
    scanProduct (assignment challengeValues batch) wasm42x6 final slot =
      scanProductAt challengeValues batch final (slot + 1) := by
  change K.mk _ _ = K.mk
    (scanProductAt challengeValues batch final (slot + 1)).c0
    (scanProductAt challengeValues batch final (slot + 1)).c1
  apply congrArg₂ K.mk
  · have low := scan_product_component_exact challengeValues batch final
        slot 0 slotBound (by decide)
    simp only [kComponentValue] at low
    rw [fieldOfNat_finVal] at low
    exact low
  · have high := scan_product_component_exact challengeValues batch final
        slot 1 slotBound (by decide)
    simp only [kComponentValue] at high
    rw [fieldOfNat_finVal] at high
    exact high

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessWords
