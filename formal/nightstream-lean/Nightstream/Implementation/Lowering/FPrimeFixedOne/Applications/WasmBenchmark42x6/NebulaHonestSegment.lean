import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaSegment

/-!
Honest two-batch Nebula execution for the modular 42-times-6 fixture.

Assurance tier: model-level.

This file binds both exact physical assignments to their source ports, proves
the public product carry between them, and derives terminal memory balance
from satisfaction of the two emitted row programs. It does not compose the
fifteen Nebula matrices with the four F-prime matrices, export a manifest,
bind Fiat--Shamir challenges, or claim a collision-probability bound.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonestSegment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaBinding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaSegment
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessWords
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonest
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

def firstBatch : Fin 2 := ⟨0, by decide⟩

def secondBatch : Fin 2 := ⟨1, by decide⟩

def firstAssignment (challengeValues : Challenges) : Nat -> F :=
  assignment challengeValues firstBatch

def secondAssignment (challengeValues : Challenges) : Nat -> F :=
  assignment challengeValues secondBatch

theorem first_ports (challengeValues : Challenges) :
    FirstPorts (firstAssignment challengeValues) := {
  constantWire := assignment_constant challengeValues firstBatch
  step := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch] using
      step_word_exact challengeValues firstBatch
  timestampIn := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch] using
      timestamp_in_word_exact challengeValues firstBatch
  timestampOut := by
    simpa [firstAssignment, Carries, fieldValue] using
      timestamp_out_word_exact challengeValues firstBatch
  pad := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch, isSecond] using
      operation_scalar_exact challengeValues firstBatch 0
  isWrite := by
    simpa [firstAssignment, Carries, fieldValue] using
      operation_scalar_exact challengeValues firstBatch 1
  ram := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch, isSecond] using
      operation_scalar_exact challengeValues firstBatch 2
  address := by
    simpa [firstAssignment, Carries, fieldValue] using
      operation_address_exact challengeValues firstBatch
  readValue := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch, isSecond] using
      operation_read_value_exact challengeValues firstBatch
  writeValue := by
    simpa [firstAssignment, Carries, fieldValue, firstBatch, isSecond] using
      operation_write_value_exact challengeValues firstBatch
  readTimestamp := by
    simpa [firstAssignment, Carries, fieldValue] using
      operation_read_timestamp_exact challengeValues firstBatch
  initialScanValue := by
    intro slot slotBound
    simpa [firstAssignment, Carries, fieldValue, firstBatch, cellAt,
      isSecond, firstCell, initialRomAt] using
      scan_value_exact challengeValues firstBatch false slot slotBound
  initialScanTimestamp := by
    intro slot slotBound
    simpa [firstAssignment, Carries, fieldValue, firstBatch, cellAt,
      isSecond, firstCell, initialRomAt] using
      scan_timestamp_exact challengeValues firstBatch false slot slotBound
  finalScanValue := by
    intro slot slotBound
    simpa [firstAssignment, Carries, fieldValue, firstBatch, cellAt,
      isSecond, firstCell, finalRomAt] using
      scan_value_exact challengeValues firstBatch true slot slotBound
  finalScanTimestamp := by
    intro slot slotBound
    simpa [firstAssignment, Carries, fieldValue, firstBatch, cellAt,
      isSecond, firstCell, finalRomAt] using
      scan_timestamp_exact challengeValues firstBatch true slot slotBound
}

theorem second_ports (challengeValues : Challenges) :
    SecondPorts (secondAssignment challengeValues) := {
  constantWire := assignment_constant challengeValues secondBatch
  step := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch] using
      step_word_exact challengeValues secondBatch
  timestampIn := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch] using
      timestamp_in_word_exact challengeValues secondBatch
  timestampOut := by
    simpa [secondAssignment, Carries, fieldValue] using
      timestamp_out_word_exact challengeValues secondBatch
  pad := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch, isSecond] using
      operation_scalar_exact challengeValues secondBatch 0
  isWrite := by
    simpa [secondAssignment, Carries, fieldValue] using
      operation_scalar_exact challengeValues secondBatch 1
  ram := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch, isSecond] using
      operation_scalar_exact challengeValues secondBatch 2
  address := by
    simpa [secondAssignment, Carries, fieldValue] using
      operation_address_exact challengeValues secondBatch
  readValue := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch, isSecond] using
      operation_read_value_exact challengeValues secondBatch
  writeValue := by
    simpa [secondAssignment, Carries, fieldValue, secondBatch, isSecond] using
      operation_write_value_exact challengeValues secondBatch
  readTimestamp := by
    simpa [secondAssignment, Carries, fieldValue] using
      operation_read_timestamp_exact challengeValues secondBatch
  initialScanValue := by
    intro slot slotBound
    simpa [secondAssignment, Carries, fieldValue, secondBatch, cellAt,
      isSecond, secondCell] using
      scan_value_exact challengeValues secondBatch false slot slotBound
  initialScanTimestamp := by
    intro slot slotBound
    simpa [secondAssignment, Carries, fieldValue, secondBatch, cellAt,
      isSecond, secondCell] using
      scan_timestamp_exact challengeValues secondBatch false slot slotBound
  finalScanValue := by
    intro slot slotBound
    simpa [secondAssignment, Carries, fieldValue, secondBatch, cellAt,
      isSecond, secondCell] using
      scan_value_exact challengeValues secondBatch true slot slotBound
  finalScanTimestamp := by
    intro slot slotBound
    simpa [secondAssignment, Carries, fieldValue, secondBatch, cellAt,
      isSecond, secondCell] using
      scan_timestamp_exact challengeValues secondBatch true slot slotBound
}

private theorem first_output_eq
    (challengeValues : Challenges) (product : Fin 4) :
    outputProductAt challengeValues firstBatch product.val =
      firstOutputProduct challengeValues product.val := by
  have alternatives : product.val = 0 ∨ product.val = 1 ∨
      product.val = 2 ∨ product.val = 3 := by omega
  rcases alternatives with value | value | value | value
  · have exact : product = ⟨0, by decide⟩ := Fin.ext value
    subst product
    rfl
  · have exact : product = ⟨1, by decide⟩ := Fin.ext value
    subst product
    rfl
  · have exact : product = ⟨2, by decide⟩ := Fin.ext value
    subst product
    rfl
  · have exact : product = ⟨3, by decide⟩ := Fin.ext value
    subst product
    rfl

theorem linked (challengeValues : Challenges) :
    Linked (firstAssignment challengeValues)
      (secondAssignment challengeValues) := {
  challengesExact := by
    rw [secondAssignment, firstAssignment,
      challenges_exact challengeValues secondBatch,
      challenges_exact challengeValues firstBatch]
  productCarry := by
    intro product
    rw [secondAssignment, firstAssignment,
      input_product_exact challengeValues secondBatch product.val
        product.isLt,
      output_product_exact challengeValues firstBatch product.val
        product.isLt]
    change firstOutputProduct challengeValues product.val =
      outputProductAt challengeValues firstBatch product.val
    exact (first_output_eq challengeValues product).symm
  initialProducts := by
    intro product
    rw [firstAssignment,
      input_product_exact challengeValues firstBatch product.val product.isLt]
    rfl
}

/-- Both exact physical programs satisfy their rows and their linked terminal
products satisfy the Nebula balance equation for the benchmark memory trace. -/
theorem two_batch_terminal_balance (challengeValues : Challenges) :
    Nightstream.Protocol.Nebula.Memory.Balanced
      (fun product =>
        outputProduct (secondAssignment challengeValues) product.val) := by
  exact final_products_balanced
    (firstAssignment challengeValues) (secondAssignment challengeValues)
    (first_ports challengeValues) (second_ports challengeValues)
    (rows_satisfied challengeValues firstBatch)
    (rows_satisfied challengeValues secondBatch)
    (linked challengeValues)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaHonestSegment
