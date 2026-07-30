import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine

/-!
Contract: attach Lean-owned canonical-u64 rows to every Poseidon2 digest lane
used by the `Pi_RLC` strong-set sampler.

Each scalar executes four digest blocks and each block exposes four field
lanes.  This module therefore emits exactly sixteen canonical-u64 occurrences
per scalar.  Their inputs are the exact symbolic output expressions of
`PiRlcCanonicalSymbolicMachine`; no digest value or generated row is imported.

This layer owns full-lane decomposition only.  It does not yet emit the
16-bit rejection candidates or first-accepted selection rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

/-- Four blocks times four field lanes. -/
def lanesPerScalar : Nat := 16

private theorem sum_const {α : Type} (items : List α) (value : Nat) :
    (items.map (fun _ => value)).sum = items.length * value := by
  rw [List.map_const', List.sum_replicate_nat]

/-- Position of an occurrence's digest block. -/
def blockOf (position : Fin lanesPerScalar) : Fin 4 :=
  ⟨position.val / 4, by
    have bounded := position.isLt
    simp only [lanesPerScalar] at bounded
    omega⟩

/-- Position of an occurrence's lane within its digest block. -/
def laneOf (position : Fin lanesPerScalar) : Fin 4 :=
  ⟨position.val % 4, Nat.mod_lt _ (by decide)⟩

/-- Symbolic state immediately before one digest block. -/
def beforeBlock
    (duplexBase : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (position : Fin lanesPerScalar) :
    SymbolicDuplex.Builder :=
  PiRlcCanonicalSymbolicMachine.stateBeforeBlock duplexBase
    (PiRlcCanonicalSymbolicMachine.enterScalar
      duplexBase builder coordinate)
    coordinate (blockOf position).val

/-- Exact symbolic Poseidon2 lane decomposed by one canonical-u64 occurrence. -/
def laneInput
    (duplexBase : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (position : Fin lanesPerScalar) :
    LinComb :=
  PiRlcCanonicalSymbolicMachine.digestLanes duplexBase
    (beforeBlock duplexBase builder coordinate position)
    (coordinate + (blockOf position).val)
    (laneOf position)

/-- Global occurrence index in a scalar batch. -/
def occurrenceIndex
    {count : Nat} (coordinate : Fin count)
    (position : Fin lanesPerScalar) : Nat :=
  coordinate.val * lanesPerScalar + position.val

/-- One canonical-u64 layout, placed in the batch's contiguous allocation. -/
def laneLayout
    (duplexBase u64Base : Nat) (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin lanesPerScalar) : CanonicalU64Recipe.Layout where
  base :=
    u64Base +
      occurrenceIndex coordinate position *
        CanonicalU64Recipe.auxiliaryCount
  input :=
    laneInput duplexBase
      (PiRlcCanonicalSymbolicMachine.stateAt
        duplexBase initial coordinate.val)
      coordinate.val position

/-- The value read by the layout is exactly the corresponding value-level
Poseidon2 digest lane. -/
theorem laneInput_eval
    (duplexBase : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder) (coordinate : Nat)
    (position : Fin lanesPerScalar)
    (valid :
      Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (beforeBlock duplexBase builder coordinate position)
          (coordinate + (blockOf position).val))) :
    lcEval assignment
        (laneInput duplexBase builder coordinate position) =
      (PiRlcCanonicalMachine.digest constants
        (PiRlcCanonicalMachine.appendRawPair constants
          (decodedBuilder assignment
            (beforeBlock duplexBase builder coordinate position))
          1 (coordinate + (blockOf position).val))).2
        (laneOf position) := by
  exact PiRlcCanonicalSymbolicMachine.digestLanes_eval
    duplexBase constants assignment constantWire
    (beforeBlock duplexBase builder coordinate position)
    (coordinate + (blockOf position).val) valid (laneOf position)

/-- Rows for one scalar's sixteen full-lane decompositions. -/
def scalarRows
    (duplexBase u64Base : Nat) (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : List Row :=
  (List.finRange lanesPerScalar).flatMap fun position =>
    CanonicalU64Recipe.rows
      (laneLayout duplexBase u64Base initial coordinate position)

/-- Rows for every scalar in a typed batch. -/
def rows
    (duplexBase u64Base count : Nat)
    (initial : SymbolicDuplex.Builder) : List Row :=
  (List.finRange count).flatMap fun coordinate =>
    scalarRows duplexBase u64Base initial coordinate

theorem scalarRows_length
    (duplexBase u64Base : Nat) (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    (scalarRows duplexBase u64Base initial coordinate).length =
      lanesPerScalar * CanonicalU64Recipe.cost.recurringRows := by
  simp [scalarRows, CanonicalU64Recipe.rows_length, lanesPerScalar,
    CanonicalU64Recipe.cost]
  rw [sum_const]
  decide

/-- Exact row count, derived from the two emitted `flatMap`s. -/
theorem rows_length
    (duplexBase u64Base count : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base count initial).length =
      count * lanesPerScalar *
        CanonicalU64Recipe.cost.recurringRows := by
  simp [rows, scalarRows_length, lanesPerScalar,
    CanonicalU64Recipe.cost]
  rw [sum_const]
  simp
  omega

theorem fixedActive_rows_length
    (duplexBase u64Base : Nat) (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base 15 initial).length = 16560 := by
  rw [rows_length]
  decide

/-- Exact contiguous allocation of the batch's canonical-u64 auxiliaries. -/
def allocation (u64Base count : Nat) : List Nat :=
  (List.range
      (count * lanesPerScalar *
        CanonicalU64Recipe.auxiliaryCount)).map
    (fun offset => u64Base + offset)

theorem allocation_length (u64Base count : Nat) :
    (allocation u64Base count).length =
      count * lanesPerScalar *
        CanonicalU64Recipe.cost.auxiliaryColumns := by
  simp [allocation, CanonicalU64Recipe.cost,
    CanonicalU64Recipe.auxiliaryCount]

theorem fixedActive_allocation_length (u64Base : Nat) :
    (allocation u64Base 15).length = 15840 := by
  rw [allocation_length]
  decide

theorem allocation_nodup (u64Base count : Nat) :
    (allocation u64Base count).Nodup := by
  unfold allocation
  exact nodup_map _ _ (fun _ _ equal => by omega) List.nodup_range

theorem allocation_nonzero
    (u64Base count column : Nat) (positive : 0 < u64Base)
    (member : column ∈ allocation u64Base count) :
    column ≠ 0 := by
  unfold allocation at member
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  omega

theorem allocation_mem_iff
    (u64Base count column : Nat) :
    column ∈ allocation u64Base count ↔
      u64Base ≤ column ∧
        column <
          u64Base + count * lanesPerScalar *
            CanonicalU64Recipe.auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have bounded := List.mem_range.mp inRange
    omega
  · intro ⟨lower, upper⟩
    apply List.mem_map.mpr
    exact ⟨column - u64Base, List.mem_range.mpr (by omega), by omega⟩

/-- Every per-lane allocation is owned by the batch allocation. -/
theorem lane_allocation_mem
    (duplexBase u64Base : Nat) (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin lanesPerScalar)
    (column : Nat)
    (member :
      column ∈
        CanonicalU64Recipe.allocation
          (laneLayout duplexBase u64Base initial coordinate position)) :
    column ∈ allocation u64Base count := by
  rw [allocation_mem_iff]
  have localWindow :=
    CanonicalU64Recipe.allocation_in_window
      (laneLayout duplexBase u64Base initial coordinate position)
      column member
  have coordinateLt := coordinate.isLt
  have positionLt := position.isLt
  simp only [laneLayout, occurrenceIndex, lanesPerScalar,
    CanonicalU64Recipe.auxiliaryCount] at localWindow ⊢
  omega

/-- Batch satisfaction restricts to any one canonical-u64 occurrence. -/
theorem satisfies_lane
    (duplexBase u64Base count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase u64Base count initial) assignment)
    (coordinate : Fin count) (position : Fin lanesPerScalar) :
    Satisfies
      (CanonicalU64Recipe.rows
        (laneLayout duplexBase u64Base initial coordinate position))
      assignment := by
  intro row rowMember
  apply satisfied row
  unfold rows scalarRows
  apply List.mem_flatMap.mpr
  refine ⟨coordinate, List.mem_finRange coordinate, ?_⟩
  apply List.mem_flatMap.mpr
  exact ⟨position, List.mem_finRange position, rowMember⟩

/-- Every satisfying batch occurrence canonically decomposes its exact
symbolic digest-lane expression. -/
theorem lane_refines
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows duplexBase u64Base count initial) assignment)
    (coordinate : Fin count) (position : Fin lanesPerScalar) :
    CanonicalU64RecipeSound.Refines assignment
      (laneLayout duplexBase u64Base initial coordinate position) := by
  apply CanonicalU64RecipeSound.sound prime canonical constantWire
  exact satisfies_lane duplexBase u64Base count initial assignment
    satisfied coordinate position

/-- Combining the canonical-u64 rows with the exact symbolic duplex
refinement identifies the represented integer with the same Poseidon2 lane. -/
theorem lane_bits_eq_digest
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base count : Nat)
    (constants : Constants) (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows duplexBase u64Base count initial) assignment)
    (coordinate : Fin count) (position : Fin lanesPerScalar)
    (valid :
      Valid duplexBase constants assignment
        (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (beforeBlock duplexBase
            (PiRlcCanonicalSymbolicMachine.stateAt
              duplexBase initial coordinate.val)
            coordinate.val position)
          (coordinate.val + (blockOf position).val))) :
    CanonicalU64RecipeSound.bitsValue assignment
        (laneLayout duplexBase u64Base initial coordinate position) =
      (PiRlcCanonicalMachine.digest constants
        (PiRlcCanonicalMachine.appendRawPair constants
          (decodedBuilder assignment
            (beforeBlock duplexBase
              (PiRlcCanonicalSymbolicMachine.stateAt
                duplexBase initial coordinate.val)
              coordinate.val position))
          1 (coordinate.val + (blockOf position).val))).2
        (laneOf position) := by
  have refined :=
    lane_refines prime duplexBase u64Base count initial canonical
      constantWire satisfied coordinate position
  rw [← refined.input_eq]
  exact laneInput_eval duplexBase constants assignment constantWire
    (PiRlcCanonicalSymbolicMachine.stateAt
      duplexBase initial coordinate.val)
    coordinate.val position valid

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64
