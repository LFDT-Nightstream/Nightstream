import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory
import Nightstream.Implementation.Lowering.Nebula.BitWitness

/-!
Exact physical witness layout for the two Nebula batches of the modular
42-times-6 WASM fixture.

Assurance tier: model-level.

This file owns the assignment function and exact access to every source and
auxiliary word. It does not claim row satisfaction, terminal balance, F-prime
placement, Rust conformance, or a security reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.BitWitness
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

private abbrev Lin := Rows.LinearCombination

/-- The selected fixture uses batch zero for ROM plus one active load and
batch one for RAM plus one padded operation slot. -/
def isSecond (batch : Fin 2) : Bool := batch.val = 1

def challengeAt (challengeValues : Challenges) : Nat -> K
  | 0 => challengeValues.gamma1
  | _ => challengeValues.gamma2

def kComponentValue (value : K) : Nat -> Nat
  | 0 => value.c0.val
  | _ => value.c1.val

def kBit (value : K) (offset : Nat) : F :=
  bitField (kComponentValue value (offset / extensionLimbBits))
    (offset % extensionLimbBits)

def kVectorBit (values : Nat -> K) (offset : Nat) : F :=
  kBit (values (offset / extensionBits)) (offset % extensionBits)

def cellBit (cell : MemTuple) (offset : Nat) : F :=
  if offset < valueBits then bitField cell.value offset
  else bitField cell.timestamp (offset - valueBits)

def cellVectorBit (cells : Nat -> MemTuple) (offset : Nat) : F :=
  cellBit (cells (offset / cellBits)) (offset % cellBits)

def productPrefix
    (challengeValues : Challenges) (cells : Nat -> MemTuple) : Nat -> K
  | 0 => K.one
  | count + 1 =>
      K.mul (productPrefix challengeValues cells count)
        (fingerprint challengeValues (cells count))

def firstCell (_final : Bool) (slot : Nat) : MemTuple :=
  Memory.blankCell slot

def secondCell (final : Bool) (slot : Nat) : MemTuple :=
  if final then Memory.finalRamAt slot else Memory.initialRamAt slot

def cellAt (batch : Fin 2) (final : Bool) (slot : Nat) : MemTuple :=
  if isSecond batch then secondCell final slot else firstCell final slot

def firstOperationProduct
    (challengeValues : Challenges) (write : Bool) : K :=
  K.mul K.one
    (fingerprint challengeValues
      (if write then Memory.writeCell else Memory.readCell))

def firstScanProduct
    (challengeValues : Challenges) (final : Bool) (count : Nat) : K :=
  K.mul K.one
    (productPrefix challengeValues (firstCell final) count)

def firstOutputProduct
    (challengeValues : Challenges) : Nat -> K
  | 0 => firstOperationProduct challengeValues false
  | 1 => firstOperationProduct challengeValues true
  | 2 => firstScanProduct challengeValues false 1024
  | _ => firstScanProduct challengeValues true 1024

def inputProductAt
    (challengeValues : Challenges) (batch : Fin 2) (product : Nat) : K :=
  if isSecond batch then firstOutputProduct challengeValues product else K.one

def operationProductAt
    (challengeValues : Challenges) (batch : Fin 2) (write : Bool) : K :=
  K.mul (inputProductAt challengeValues batch (if write then 1 else 0))
    (if isSecond batch then K.one
      else fingerprint challengeValues
        (if write then Memory.writeCell else Memory.readCell))

def scanProductAt
    (challengeValues : Challenges) (batch : Fin 2)
    (final : Bool) (count : Nat) : K :=
  K.mul (inputProductAt challengeValues batch (if final then 3 else 2))
    (productPrefix challengeValues (cellAt batch final) count)

def outputProductAt
    (challengeValues : Challenges) (batch : Fin 2) : Nat -> K
  | 0 => operationProductAt challengeValues batch false
  | 1 => operationProductAt challengeValues batch true
  | 2 => scanProductAt challengeValues batch false 1024
  | _ => scanProductAt challengeValues batch true 1024

def publicBit
    (challengeValues : Challenges) (batch : Fin 2) (offset : Nat) : F :=
  if offset < XOffset.step then
    bitField 0 offset
  else if offset < XOffset.timestampIn then
    bitField batch.val (offset - XOffset.step)
  else if offset < XOffset.timestampOut then
    bitField batch.val (offset - XOffset.timestampIn)
  else if offset < XOffset.gamma then
    bitField 1 (offset - XOffset.timestampOut)
  else if offset < XOffset.productsIn then
    kVectorBit (challengeAt challengeValues) (offset - XOffset.gamma)
  else if offset < XOffset.productsOut then
    kVectorBit (inputProductAt challengeValues batch)
      (offset - XOffset.productsIn)
  else
    kVectorBit (outputProductAt challengeValues batch)
      (offset - XOffset.productsOut)

def operationBit (batch : Fin 2) (offset : Nat) : F :=
  if offset = 0 then bitField (if isSecond batch then 1 else 0) 0
  else if offset = 1 then bitField 0 0
  else if offset = 2 then bitField (if isSecond batch then 0 else 1) 0
  else if offset < 3 + wasm42x6.addressBits then
    bitField 0 (offset - 3)
  else if offset < 3 + wasm42x6.addressBits + valueBits then
    bitField (if isSecond batch then 0 else 42)
      (offset - (3 + wasm42x6.addressBits))
  else if offset < 3 + wasm42x6.addressBits + 2 * valueBits then
    bitField (if isSecond batch then 0 else 42)
      (offset - (3 + wasm42x6.addressBits + valueBits))
  else
    bitField 0
      (offset - (3 + wasm42x6.addressBits + 2 * valueBits))

def operationAuxiliaryBit
    (challengeValues : Challenges) (batch : Fin 2) (offset : Nat) : F :=
  if offset < Layout.timestampBits then
    bitField 0 offset
  else if offset < Layout.timestampBits + wasm42x6.countBits then
    bitField (if isSecond batch then 0 else 1)
      (offset - Layout.timestampBits)
  else if offset < Layout.timestampBits + wasm42x6.countBits +
      extensionBits then
    kBit (operationProductAt challengeValues batch false)
      (offset - (Layout.timestampBits + wasm42x6.countBits))
  else
    kBit (operationProductAt challengeValues batch true)
      (offset -
        (Layout.timestampBits + wasm42x6.countBits + extensionBits))

def scanAuxiliaryBit
    (challengeValues : Challenges) (batch : Fin 2) (offset : Nat) : F :=
  let slot := offset / scanAuxiliaryBits
  let localOffset := offset % scanAuxiliaryBits
  if localOffset < extensionBits then
    kBit (scanProductAt challengeValues batch false (slot + 1)) localOffset
  else
    kBit (scanProductAt challengeValues batch true (slot + 1))
      (localOffset - extensionBits)

/-- Exact logical assignment for one selected Nebula batch. Columns outside
the selected layout are zero. -/
def assignment (challengeValues : Challenges) (batch : Fin 2) : Nat -> F :=
  fun column =>
    if column = 0 then 1
    else if column < wasm42x6.publicEnd then
      publicBit challengeValues batch (column - 1)
    else if column < wasm42x6.operationLane then 0
    else if column < wasm42x6.operationLane + wasm42x6.operationBits then
      operationBit batch (column - wasm42x6.operationLane)
    else if column < wasm42x6.initialScanLane then 0
    else if column < wasm42x6.initialScanLane +
        wasm42x6.scanSlots * cellBits then
      cellVectorBit (cellAt batch false)
        (column - wasm42x6.initialScanLane)
    else if column < wasm42x6.finalScanLane then 0
    else if column < wasm42x6.finalScanLane +
        wasm42x6.scanSlots * cellBits then
      cellVectorBit (cellAt batch true)
        (column - wasm42x6.finalScanLane)
    else if column < wasm42x6.auxiliaryStart then 0
    else if column < wasm42x6.scanAuxiliaryStart then
      operationAuxiliaryBit challengeValues batch
        (column - wasm42x6.auxiliaryStart)
    else if column < wasm42x6.columnCount then
      scanAuxiliaryBit challengeValues batch
        (column - wasm42x6.scanAuxiliaryStart)
    else 0

@[simp] theorem assignment_constant
    (challengeValues : Challenges) (batch : Fin 2) :
    assignment challengeValues batch 0 = 1 := by
  simp [assignment]

theorem kBit_at
    (value : K) (component bit : Nat)
    (_componentBound : component < 2) (bitBound : bit < extensionLimbBits) :
    kBit value (component * extensionLimbBits + bit) =
      bitField (kComponentValue value component) bit := by
  have positive : 0 < extensionLimbBits := by decide
  have divExact :
      (component * extensionLimbBits + bit) / extensionLimbBits =
        component := by
    rw [Nat.mul_comm component extensionLimbBits,
      Nat.mul_add_div positive, Nat.div_eq_of_lt bitBound]
    omega
  have modExact :
      (component * extensionLimbBits + bit) % extensionLimbBits = bit :=
    Nat.mul_add_mod_of_lt bitBound
  simp [kBit, divExact, modExact]

theorem kVectorBit_at
    (values : Nat -> K) (index component bit : Nat)
    (componentBound : component < 2) (bitBound : bit < extensionLimbBits) :
    kVectorBit values
        (index * extensionBits + component * extensionLimbBits + bit) =
      bitField (kComponentValue (values index) component) bit := by
  have localBound : component * extensionLimbBits + bit < extensionBits := by
    simp [extensionBits, extensionLimbBits] at componentBound bitBound ⊢
    omega
  have positive : 0 < extensionBits := by decide
  have divExact :
      (index * extensionBits +
          (component * extensionLimbBits + bit)) / extensionBits = index := by
    rw [Nat.mul_comm index extensionBits,
      Nat.mul_add_div positive, Nat.div_eq_of_lt localBound]
    omega
  have modExact :
      (index * extensionBits +
          (component * extensionLimbBits + bit)) % extensionBits =
        component * extensionLimbBits + bit :=
    Nat.mul_add_mod_of_lt localBound
  rw [kVectorBit, show
      index * extensionBits + component * extensionLimbBits + bit =
        index * extensionBits + (component * extensionLimbBits + bit) by
      omega, divExact, modExact]
  exact kBit_at (values index) component bit componentBound bitBound

theorem cellVectorBit_at
    (cells : Nat -> MemTuple) (slot offset : Nat)
    (offsetBound : offset < cellBits) :
    cellVectorBit cells (slot * cellBits + offset) =
      cellBit (cells slot) offset := by
  have positive : 0 < cellBits := by decide
  have divExact : (slot * cellBits + offset) / cellBits = slot := by
    rw [Nat.mul_comm slot cellBits,
      Nat.mul_add_div positive, Nat.div_eq_of_lt offsetBound]
    omega
  have modExact : (slot * cellBits + offset) % cellBits = offset :=
    Nat.mul_add_mod_of_lt offsetBound
  rw [cellVectorBit, divExact, modExact]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaWitnessLayout
