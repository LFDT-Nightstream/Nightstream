import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64

/-!
Contract: compose the 64 Lean-owned candidate-classification occurrences for
every `Pi_RLC` scalar.

The source order is fixed here, independently of Rust:

```
candidate = 16 * block + 4 * lane + part
```

Each source is the exact 16-bit slice of the corresponding canonical-u64
digest lane.  Candidate zero reads the literal empty/zero prefix; every later
candidate reads the immediately preceding cumulative column.

This file owns construction, exact row/column cost, placement, and restriction
to one occurrence.  Batch semantic induction and honest witness threading are
separate proof responsibilities.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def candidatesPerScalar : Nat := 64

structure Address where
  block : Fin 4
  lane : Fin 4
  part : Fin 4

def address (candidate : Fin candidatesPerScalar) : Address where
  block :=
    ⟨candidate.val / 16, by
      have bounded := candidate.isLt
      simp only [candidatesPerScalar] at bounded
      omega⟩
  lane :=
    ⟨(candidate.val % 16) / 4, by
      have remainder := Nat.mod_lt candidate.val (by decide : 0 < 16)
      omega⟩
  part :=
    ⟨candidate.val % 4, Nat.mod_lt _ (by decide)⟩

theorem address_recomposes (candidate : Fin candidatesPerScalar) :
    16 * (address candidate).block.val +
        4 * (address candidate).lane.val +
        (address candidate).part.val =
      candidate.val := by
  have outer := Nat.div_add_mod candidate.val 16
  have inner := Nat.div_add_mod (candidate.val % 16) 4
  simp only [address]
  omega

def lanePosition (candidate : Fin candidatesPerScalar) :
    Fin PiRlcCanonicalU64.lanesPerScalar :=
  ⟨(address candidate).block.val * 4 + (address candidate).lane.val, by
    have blockLt := (address candidate).block.isLt
    have laneLt := (address candidate).lane.isLt
    simp only [PiRlcCanonicalU64.lanesPerScalar]
    omega⟩

def sourceBitIndex
    (candidate : Fin candidatesPerScalar)
    (bit : Fin PiRlcCanonicalCandidate.sourceBitCount) : Nat :=
  (address candidate).part.val * 16 + bit.val

theorem sourceBitIndex_lt
    (candidate : Fin candidatesPerScalar)
    (bit : Fin PiRlcCanonicalCandidate.sourceBitCount) :
    sourceBitIndex candidate bit < 64 := by
  have partLt := (address candidate).part.isLt
  have bitLt := bit.isLt
  simp only [sourceBitIndex,
    PiRlcCanonicalCandidate.sourceBitCount] at bitLt ⊢
  omega

def occurrenceIndex
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : Nat :=
  coordinate.val * candidatesPerScalar + candidate.val

def occurrenceBase
    (candidateBase : Nat)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : Nat :=
  candidateBase +
    occurrenceIndex coordinate candidate *
      PiRlcCanonicalCandidate.auxiliaryCount

def prior
    (candidateBase : Nat)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : LinComb :=
  if candidate.val = 0 then []
  else
    [(occurrenceBase candidateBase coordinate candidate - 1, 1)]

/-- One occurrence wired to the exact canonical-u64 source slice. -/
def candidateLayout
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalCandidate.Layout where
  base := occurrenceBase candidateBase coordinate candidate
  sourceBit := fun bit =>
    CanonicalU64Recipe.bitColumn
      (PiRlcCanonicalU64.laneLayout duplexBase u64Base initial coordinate
        (lanePosition candidate))
      (sourceBitIndex candidate bit)
  prior := prior candidateBase coordinate candidate

theorem prior_zero
    (candidateBase : Nat) {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) (zero : candidate.val = 0) :
    prior candidateBase coordinate candidate = [] := by
  simp [prior, zero]

theorem prior_successor
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) (positive : 0 < candidate.val) :
    prior candidateBase coordinate candidate =
      [(PiRlcCanonicalCandidate.cumulativeColumn
        (candidateLayout duplexBase u64Base candidateBase initial
          coordinate
          ⟨candidate.val - 1, by
            have bounded := candidate.isLt
            change candidate.val < 64 at bounded
            change candidate.val - 1 < 64
            omega⟩), 1)] := by
  have nonzero : candidate.val ≠ 0 := by omega
  simp only [prior, nonzero, if_false]
  congr 2
  simp only [PiRlcCanonicalCandidate.cumulativeColumn, candidateLayout,
    occurrenceBase, occurrenceIndex,
    PiRlcCanonicalCandidate.auxiliaryCount, candidatesPerScalar]
  omega

/-- Rows for one scalar's 64 candidate occurrences. -/
def scalarRows
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : List Row :=
  (List.finRange candidatesPerScalar).flatMap fun candidate =>
    PiRlcCanonicalCandidate.rows
      (candidateLayout duplexBase u64Base candidateBase initial
        coordinate candidate)

/-- Rows for all scalar coordinates. -/
def rows
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder) : List Row :=
  (List.finRange count).flatMap fun coordinate =>
    scalarRows duplexBase u64Base candidateBase initial coordinate

private theorem sum_const {α : Type} (items : List α) (value : Nat) :
    (items.map (fun _ => value)).sum = items.length * value := by
  rw [List.map_const', List.sum_replicate_nat]

theorem scalarRows_length
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    (scalarRows duplexBase u64Base candidateBase initial coordinate).length =
      candidatesPerScalar *
        PiRlcCanonicalCandidate.cost.recurringRows := by
  simp [scalarRows, PiRlcCanonicalCandidate.rows_length,
    candidatesPerScalar, PiRlcCanonicalCandidate.cost]
  rw [sum_const]
  decide

theorem rows_length
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base candidateBase count initial).length =
      count * candidatesPerScalar *
        PiRlcCanonicalCandidate.cost.recurringRows := by
  simp [rows, scalarRows_length, candidatesPerScalar,
    PiRlcCanonicalCandidate.cost]
  rw [sum_const]
  simp
  omega

theorem fixedActive_rows_length
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base candidateBase 15 initial).length = 24000 := by
  rw [rows_length]
  decide

def allocation (candidateBase count : Nat) : List Nat :=
  (List.range
      (count * candidatesPerScalar *
        PiRlcCanonicalCandidate.auxiliaryCount)).map
    (fun offset => candidateBase + offset)

theorem allocation_length (candidateBase count : Nat) :
    (allocation candidateBase count).length =
      count * candidatesPerScalar *
        PiRlcCanonicalCandidate.cost.auxiliaryColumns := by
  simp [allocation, PiRlcCanonicalCandidate.cost,
    PiRlcCanonicalCandidate.auxiliaryCount]

theorem fixedActive_allocation_length (candidateBase : Nat) :
    (allocation candidateBase 15).length = 21120 := by
  rw [allocation_length]
  decide

theorem allocation_nodup (candidateBase count : Nat) :
    (allocation candidateBase count).Nodup := by
  unfold allocation
  exact nodup_map _ _ (fun _ _ equal => by omega) List.nodup_range

theorem allocation_nonzero
    (candidateBase count column : Nat) (positive : 0 < candidateBase)
    (member : column ∈ allocation candidateBase count) :
    column ≠ 0 := by
  unfold allocation at member
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  omega

theorem allocation_mem_iff
    (candidateBase count column : Nat) :
    column ∈ allocation candidateBase count ↔
      candidateBase ≤ column ∧
        column <
          candidateBase + count * candidatesPerScalar *
            PiRlcCanonicalCandidate.auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have bounded := List.mem_range.mp inRange
    omega
  · intro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - candidateBase, List.mem_range.mpr (by omega), by omega⟩

theorem occurrence_allocation_mem
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar)
    (column : Nat)
    (member :
      column ∈ PiRlcCanonicalCandidate.allocation
        (candidateLayout duplexBase u64Base candidateBase initial
          coordinate candidate)) :
    column ∈ allocation candidateBase count := by
  rw [allocation_mem_iff]
  have localWindow :=
    (PiRlcCanonicalCandidate.allocation_mem_iff _ _).mp member
  have coordinateLt := coordinate.isLt
  have candidateLt := candidate.isLt
  simp only [candidateLayout, occurrenceBase, occurrenceIndex,
    candidatesPerScalar, PiRlcCanonicalCandidate.auxiliaryCount] at localWindow ⊢
  omega

theorem satisfies_candidate
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase count initial) assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    Satisfies
      (PiRlcCanonicalCandidate.rows
        (candidateLayout duplexBase u64Base candidateBase initial
          coordinate candidate))
      assignment := by
  intro row rowMember
  apply satisfied row
  unfold rows scalarRows
  apply List.mem_flatMap.mpr
  refine ⟨coordinate, List.mem_finRange coordinate, ?_⟩
  apply List.mem_flatMap.mpr
  exact ⟨candidate, List.mem_finRange candidate, rowMember⟩

theorem sourceBitsBoolean
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalCandidateSound.SourceBitsBoolean assignment
      (candidateLayout duplexBase u64Base candidateBase initial
        coordinate candidate) := by
  intro bit
  have refined :=
    PiRlcCanonicalU64.lane_refines prime duplexBase u64Base count initial
      canonical constantWire u64Satisfied coordinate (lanePosition candidate)
  exact refined.bit (sourceBitIndex candidate bit)
    (sourceBitIndex_lt candidate bit)

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
