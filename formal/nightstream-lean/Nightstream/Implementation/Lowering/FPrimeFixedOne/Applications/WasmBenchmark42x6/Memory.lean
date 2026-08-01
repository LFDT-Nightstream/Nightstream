import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Semantics
import Nightstream.Protocol.Nebula.Memory

/-!
Contract: exact stackless memory trace for the modular 42-times-6 fixture.

Assurance tier: model-level.

The first batch reads RAM address zero and writes the same value back at the
next timestamp. The second batch has no memory access. The two 1,024-cell scan
chunks cover the selected 1,024 ROM cells followed by 1,024 RAM cells.

This file owns the protocol trace. It does not own physical columns, a CCS
assignment, transcript challenges, F-prime assembly, Rust, or a security
reduction.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

def romCells : Nat := 1024
def ramCells : Nat := 1024
def loadedGlobalIndex : Nat := romCells

def blankCell (globalIndex : Nat) : MemTuple where
  timestamp := 0
  globalIndex := globalIndex
  value := 0

def readCell : MemTuple where
  timestamp := 0
  globalIndex := loadedGlobalIndex
  value := 42

def writeCell : MemTuple where
  timestamp := 1
  globalIndex := loadedGlobalIndex
  value := 42

def initialRamAt (slot : Nat) : MemTuple :=
  if slot = 0 then readCell
  else blankCell (loadedGlobalIndex + slot)

def finalRamAt (slot : Nat) : MemTuple :=
  if slot = 0 then writeCell
  else blankCell (loadedGlobalIndex + slot)

def access : Nightstream.Protocol.Nebula.Memory.Access where
  read := readCell
  write := writeCell

/-- The first scan chunk covers the ROM namespace. The benchmark memory
relation does not read or write this namespace. -/
def romChunk : List MemTuple :=
  (List.range romCells).map blankCell

/-- Cells after RAM address zero. -/
def ramTail : List MemTuple :=
  (List.range (ramCells - 1)).map fun offset =>
    blankCell (loadedGlobalIndex + 1 + offset)

def initialRamChunk : List MemTuple := readCell :: ramTail
def finalRamChunk : List MemTuple := writeCell :: ramTail

def initialSnapshot : List MemTuple := romChunk ++ initialRamChunk
def finalSnapshot : List MemTuple := romChunk ++ finalRamChunk

private theorem initialRamPrefix : forall count,
    (List.range (count + 1)).map initialRamAt =
      readCell ::
        (List.range count).map fun offset =>
          blankCell (loadedGlobalIndex + 1 + offset) := by
  intro count
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append, inductionHypothesis,
        List.range_succ, List.map_append]
      simp [initialRamAt, Nat.add_assoc, Nat.add_comm count 1]

private theorem finalRamPrefix : forall count,
    (List.range (count + 1)).map finalRamAt =
      writeCell ::
        (List.range count).map fun offset =>
          blankCell (loadedGlobalIndex + 1 + offset) := by
  intro count
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append, inductionHypothesis,
        List.range_succ, List.map_append]
      simp [finalRamAt, Nat.add_assoc, Nat.add_comm count 1]

theorem initialRamChunk_eq_map :
    initialRamChunk = (List.range 1024).map initialRamAt := by
  rw [show 1024 = 1023 + 1 by decide, initialRamPrefix]
  rfl

theorem finalRamChunk_eq_map :
    finalRamChunk = (List.range 1024).map finalRamAt := by
  rw [show 1024 = 1023 + 1 by decide, finalRamPrefix]
  rfl

theorem namespace_sizes :
    romChunk.length = 1024 ∧
      initialRamChunk.length = 1024 ∧
      finalRamChunk.length = 1024 := by
  simp [romChunk, initialRamChunk, finalRamChunk, ramTail,
    romCells, ramCells]

def accessApplies :
    Nightstream.Protocol.Nebula.Memory.Applies
      initialSnapshot 0 access finalSnapshot 1 := by
  refine
    { left := romChunk
      right := ramTail
      beforeExact := rfl
      afterExact := rfl
      sameCell := rfl
      previousTimestamp := by decide
      writeTimestamp := rfl
      timestampExact := rfl }

theorem execution :
    Nightstream.Protocol.Nebula.Memory.Executes
      initialSnapshot 0 [access] finalSnapshot 1 := by
  exact .cons accessApplies (.nil finalSnapshot 1)

/-- The exact benchmark memory trace satisfies Nebula's terminal multiset
equation for every verifier challenge. -/
theorem balanced (challengeValues : Challenges) :
    Nightstream.Protocol.Nebula.Memory.Balanced
      (Nightstream.Protocol.Nebula.Memory.products challengeValues
        initialSnapshot [access] finalSnapshot) :=
  Nightstream.Protocol.Nebula.Memory.executes_balanced challengeValues execution

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory
