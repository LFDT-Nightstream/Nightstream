import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound

/-!
Contract: typed, activation-compatible occurrence of the exact production
Goldilocks Poseidon2 width-eight permutation.

Assurance tier: artifact-checked.

Owns:
- the exact `0..608` numeric-to-typed column map for one isolated
  permutation occurrence;
- 600 internal SSA rows translated without semantic loss;
- eight activation-gated copies from internal permutation outputs to visible
  outputs;
- exact row/temporary cost and a nonoptional receipt;
- active soundness, honest active completion, and inactive completion.

Does not own: sponge absorption, padding, domain separation, XOut
serialization, the optional hash wrapper, either hash `CallRecipe`,
generated call-site placement, native Poseidon2 parity, or collision
resistance.

Emits constraints: exactly 608 rows and 600 auxiliary temporary columns.
The eight visible outputs are allocated by the enclosing call receipt.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe

set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace NumericPoseidon

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

abbrev rows :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.rows

abbrev definitions :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions

abbrev interpret :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret

abbrev permute :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute

abbrev inputOnly :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly

abbrev permuteState :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permuteState

abbrev permutationAssignment :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment

abbrev inputColumns :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns

abbrev outputColumns :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns

end NumericPoseidon

def inputWidth : Nat := 8

def outputWidth : Nat := 8

def temporaryWidth : Nat := 600

def coreRowCount : Nat := 600

def gateRowCount : Nat := 8

def recurringRows : Nat := coreRowCount + gateRowCount

def inputLayout : Layout :=
  auxiliaryLayout inputWidth

def outputLayout : Layout :=
  auxiliaryLayout outputWidth

def temporaryLayout : Layout :=
  auxiliaryLayout temporaryWidth

/-- The exact physical data supplied by one permutation occurrence. Outputs
and temporaries are fresh allocations; only temporaries may be changed by
completion. -/
structure Frame where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  active : ColumnId
  input : ColumnBundle inputLayout
  output : ColumnBundle outputLayout
  temporaries : ColumnBundle temporaryLayout
  allocationsNodup :
    (output.ids ++ temporaries.ids).Nodup
  temporariesDisjointVisible :
    IdsDisjoint temporaries.ids
      ([one, active] ++ input.ids ++ output.ids)
  outputsDisjointPreexisting :
    IdsDisjoint output.ids ([one, active] ++ input.ids)
  allocationsOwned :
    ∀ column,
      column ∈ output.columns ++ temporaries.columns ->
        column.id.owner = owner

namespace Frame

def visibleIds (frame : Frame) : List ColumnId :=
  [frame.one, frame.active] ++ frame.input.ids ++ frame.output.ids

def allocations (frame : Frame) : List OwnedColumn :=
  frame.output.columns ++ frame.temporaries.columns

@[simp] theorem input_ids_length (frame : Frame) :
    frame.input.ids.length = inputWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.input.length_eq]
  simp [inputLayout, inputWidth, auxiliaryLayout, ownedLayout]

@[simp] theorem output_ids_length (frame : Frame) :
    frame.output.ids.length = outputWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.output.length_eq]
  simp [outputLayout, outputWidth, auxiliaryLayout, ownedLayout]

@[simp] theorem temporary_ids_length (frame : Frame) :
    frame.temporaries.ids.length = temporaryWidth := by
  rw [ColumnBundle.ids, List.length_map, frame.temporaries.length_eq]
  change
    (List.replicate temporaryWidth Ownership.auxiliaryColumn).length =
      temporaryWidth
  exact List.length_replicate ..

end Frame

def inputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  frame.input.ids.getD lane frame.one

def outputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  frame.output.ids.getD lane frame.one

def temporaryColumn (frame : Frame) (index : Nat) : ColumnId :=
  frame.temporaries.ids.getD index frame.one

/-- Artifact source column 0 is constant one, columns 1..8 are visible
inputs, and columns 9..608 are the 600 internal SSA temporaries. -/
def columnMap (frame : Frame) (source : Nat) : ColumnId :=
  if source = 0 then frame.one
  else if source < 9 then inputColumn frame (source - 1)
  else temporaryColumn frame (source - 9)

/-- The artifact outputs 601..608 are temporary coordinates 592..599. -/
def internalOutputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  temporaryColumn frame (592 + lane)

@[simp] theorem columnMap_zero (frame : Frame) :
    columnMap frame 0 = frame.one := by
  simp [columnMap]

theorem columnMap_input (frame : Frame) (lane : Nat)
    (laneLt : lane < inputWidth) :
    columnMap frame (lane + 1) = inputColumn frame lane := by
  unfold inputWidth at laneLt
  unfold columnMap
  have nonzero : lane + 1 ≠ 0 := by omega
  have below : lane + 1 < 9 := by omega
  rw [if_neg nonzero, if_pos below]
  congr 1

theorem columnMap_output (frame : Frame) (lane : Nat) :
    columnMap frame (601 + lane) = internalOutputColumn frame lane := by
  unfold columnMap internalOutputColumn
  have nonzero : 601 + lane ≠ 0 := by omega
  have notBelow : ¬ 601 + lane < 9 := by omega
  rw [if_neg nonzero, if_neg notBelow]
  congr 1
  omega

def coreRows (frame : Frame) : List OwnedRow :=
  ownedRowsFrom frame.owner frame.firstOrdinal (columnMap frame)
    NumericPoseidon.rows

def gateRow (frame : Frame) (lane : Nat) : OwnedRow where
  id := {
    owner := frame.owner
    ordinal := frame.firstOrdinal + coreRowCount + lane
  }
  row := {
    a := singleton frame.active 1
    b := difference
      (internalOutputColumn frame lane)
      (outputColumn frame lane)
    c := []
  }

def gateRows (frame : Frame) : List OwnedRow :=
  (List.range gateRowCount).map (gateRow frame)

def rows (frame : Frame) : List OwnedRow :=
  coreRows frame ++ gateRows frame

/-- Program-derived physical footprint of one activation-compatible
permutation occurrence. -/
def footprint : CallFootprint where
  recurringRows := recurringRows
  temporaries := [temporaryLayout]

/-- Mandatory physical emission receipt. No row or allocation is supplied
outside the exact visible-output, internal-temporary, and row lists. -/
def receipt (frame : Frame) : CallReceipt where
  outputBundles := [frame.output.columns]
  temporaryBundles := [frame.temporaries.columns]
  rows := rows frame

theorem receipt_exact (frame : Frame) :
    receipt frame =
      { outputBundles := [frame.output.columns]
        temporaryBundles := [frame.temporaries.columns]
        rows := rows frame } :=
  rfl

theorem coreRows_length (frame : Frame) :
    (coreRows frame).length = coreRowCount := by
  rw [coreRows, ownedRowsFrom_length]
  exact
    Nightstream.Implementation.R1CS.Poseidon2Permutation.rows_length

theorem gateRows_length (frame : Frame) :
    (gateRows frame).length = gateRowCount := by
  simp [gateRows]

theorem rows_length (frame : Frame) :
    (rows frame).length = recurringRows := by
  rw [rows, List.length_append, coreRows_length, gateRows_length]
  rfl

theorem receipt_row_count (frame : Frame) :
    (receipt frame).rows.length = footprint.recurringRows := by
  exact rows_length frame

theorem rows_owned
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_append.mp member with coreMember | gateMember
  · exact ownedRowsFrom_owned frame.owner frame.firstOrdinal
      (columnMap frame) NumericPoseidon.rows owned coreMember
  · rcases List.mem_map.mp gateMember with ⟨lane, laneMember, equal⟩
    subst owned
    rfl

private theorem gateIds_nodup_of
    (frame : Frame)
    (lanes : List Nat)
    (nodup : lanes.Nodup) :
    ((lanes.map (gateRow frame)).map (fun row => row.id)).Nodup := by
  rw [List.map_map]
  change
    (lanes.map (fun lane => (gateRow frame lane).id)).Nodup
  induction lanes with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have split : head ∉ tail ∧ tail.Nodup := by
        simpa only [List.nodup_cons] using nodup
      rw [List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.mp member with
          ⟨lane, laneMember, equal⟩
        have ordinalEqual := congrArg RowId.ordinal equal
        simp only [gateRow] at ordinalEqual
        have laneEqual : lane = head := by
          omega
        exact split.1 (laneEqual ▸ laneMember)
      · exact inductionHypothesis split.2

private theorem gateRows_ids_nodup (frame : Frame) :
    ((gateRows frame).map (fun row => row.id)).Nodup := by
  exact gateIds_nodup_of frame (List.range gateRowCount)
    List.nodup_range

private theorem coreRow_ordinal_lt
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ coreRows frame) :
    owned.id.ordinal < frame.firstOrdinal + coreRowCount := by
  have mappedMember :
      owned.id ∈ (coreRows frame).map (fun row => row.id) :=
    List.mem_map.mpr ⟨owned, member, rfl⟩
  rw [coreRows,
    ownedRowsFrom_ids_exact frame.owner frame.firstOrdinal
      (columnMap frame) NumericPoseidon.rows] at mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨ordinal, ordinalMember, equal⟩
  rcases List.mem_range'.mp ordinalMember with
    ⟨offset, offsetLt, ordinalEqual⟩
  have ordinalLt :
      ordinal < frame.firstOrdinal + NumericPoseidon.rows.length := by
    simp only [Nat.one_mul] at ordinalEqual
    omega
  have exactOrdinal : ordinal = owned.id.ordinal := by
    simpa using congrArg RowId.ordinal equal
  have exactLength :
      NumericPoseidon.rows.length = coreRowCount := by
    simpa [coreRowCount,
      Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount] using
      Nightstream.Implementation.R1CS.Poseidon2Permutation.rows_length
  calc
    owned.id.ordinal = ordinal := exactOrdinal.symm
    _ < frame.firstOrdinal + NumericPoseidon.rows.length := ordinalLt
    _ = frame.firstOrdinal + coreRowCount := by rw [exactLength]

private theorem gateRow_ordinal_ge
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ gateRows frame) :
    frame.firstOrdinal + coreRowCount ≤ owned.id.ordinal := by
  rcases List.mem_map.mp member with ⟨lane, laneMember, equal⟩
  subst owned
  simp [gateRow]

theorem rowIds_nodup (frame : Frame) :
    ((rows frame).map (fun row => row.id)).Nodup := by
  rw [rows, List.map_append, List.nodup_append]
  refine ⟨
    ownedRowsFrom_ids_nodup frame.owner frame.firstOrdinal
      (columnMap frame) NumericPoseidon.rows,
    gateRows_ids_nodup frame,
    ?_⟩
  intro coreId coreMember gateId gateMember equal
  rcases List.mem_map.mp coreMember with
    ⟨coreRow, coreRowMember, coreEqual⟩
  rcases List.mem_map.mp gateMember with
    ⟨gate, gateRowMember, gateEqual⟩
  have below := coreRow_ordinal_lt frame coreRow coreRowMember
  have above := gateRow_ordinal_ge frame gate gateRowMember
  have rowIdsEqual : coreRow.id = gate.id :=
    coreEqual.trans (equal.trans gateEqual.symm)
  have ordinalEqual := congrArg RowId.ordinal rowIdsEqual
  omega

/-! ## Executable permutation semantics -/

/-- Canonical numeric input lane read from the visible typed assignment. -/
def inputLaneValue
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (lane : Nat) : Nat :=
  (assignment (inputColumn frame lane)).val

/-- Pull every numeric source column through this occurrence's exact map. -/
def initialNumeric
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    Nat -> Nat :=
  numericAssignment (columnMap frame) assignment

/-- Execute the exact 600-definition production permutation program. -/
def execution
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    Nat -> Nat :=
  NumericPoseidon.interpret (initialNumeric frame assignment)

/-- Visible field semantics of one width-eight output lane. -/
def semanticLane
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (lane : Nat) : Field :=
  residue
    (NumericPoseidon.permute (inputLaneValue frame assignment) lane)

theorem initialNumeric_canonical
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (source : Nat) :
    initialNumeric frame assignment source < Numeric.modulus := by
  exact numericAssignment_canonical (columnMap frame) assignment source

theorem initialNumeric_zero
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1) :
    initialNumeric frame assignment 0 = 1 := by
  change (assignment (columnMap frame 0)).val = 1
  rw [columnMap_zero, constantOne]
  rfl

theorem initialNumeric_input
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (lane : Nat)
    (laneLt : lane < inputWidth) :
    initialNumeric frame assignment (lane + 1) =
      inputLaneValue frame assignment lane := by
  rw [initialNumeric, numericAssignment,
    columnMap_input frame lane laneLt]
  rfl

private theorem output_source_mem
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    601 + lane ∈ NumericPoseidon.outputColumns := by
  unfold outputWidth at laneLt
  have cases :
      lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 ∨
        lane = 4 ∨ lane = 5 ∨ lane = 6 ∨ lane = 7 := by
    omega
  rcases cases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [
      Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns
    ]

private theorem input_source_mem
    (source : Nat)
    (sourceLt : source < 9) :
    source ∈ NumericPoseidon.inputColumns := by
  have cases :
      source = 0 ∨ source = 1 ∨ source = 2 ∨ source = 3 ∨
        source = 4 ∨ source = 5 ∨ source = 6 ∨ source = 7 ∨
          source = 8 := by
    omega
  rcases cases with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns
      ]

theorem permuteState_initial_eq_permute
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (lane : Nat) :
    NumericPoseidon.permuteState (initialNumeric frame assignment)
        (601 + lane) =
      NumericPoseidon.permute (inputLaneValue frame assignment) lane := by
  have inputOnlyEqual :
      Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
          (initialNumeric frame assignment) =
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
          (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment
            (inputLaneValue frame assignment)) := by
    funext source
    unfold
      Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
    by_cases sourceLt : source < 9
    · rw [if_pos sourceLt, if_pos sourceLt]
      by_cases sourceZero : source = 0
      · subst source
        simp [
          Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment,
          initialNumeric_zero frame assignment constantOne
        ]
      · have sourcePositive : 0 < source :=
          Nat.pos_of_ne_zero sourceZero
        have laneLt : source - 1 < inputWidth := by
          unfold inputWidth
          omega
        have sourceEq : source - 1 + 1 = source := by
          omega
        rw [← sourceEq,
          initialNumeric_input frame assignment (source - 1) laneLt]
        simp [
          Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment,
          inputLaneValue
        ]
    · rw [if_neg sourceLt, if_neg sourceLt]
  calc
    NumericPoseidon.permuteState (initialNumeric frame assignment)
          (601 + lane) =
        NumericPoseidon.permuteState
      (NumericPoseidon.permutationAssignment
            (inputLaneValue frame assignment))
          (601 + lane) := by
      change
        NumericPoseidon.interpret
            (NumericPoseidon.inputOnly
              (initialNumeric frame assignment))
            (601 + lane) =
          NumericPoseidon.interpret
            (NumericPoseidon.inputOnly
              (NumericPoseidon.permutationAssignment
                (inputLaneValue frame assignment)))
            (601 + lane)
      exact congrArg
        (fun state => NumericPoseidon.interpret state (601 + lane))
        inputOnlyEqual
    _ = NumericPoseidon.permute
          (inputLaneValue frame assignment) lane :=
      (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_eq
        (inputLaneValue frame assignment) lane).symm

theorem execution_output_eq_semantic
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    residue (execution frame assignment (601 + lane)) =
      semanticLane frame assignment lane := by
  have interpreted :=
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret_output_eq_permuteState
      (initialNumeric frame assignment) (601 + lane)
      (output_source_mem lane laneLt)
  change
    residue
        (NumericPoseidon.interpret
          (initialNumeric frame assignment) (601 + lane)) =
      residue
        (NumericPoseidon.permute
          (inputLaneValue frame assignment) lane)
  calc
    residue (execution frame assignment (601 + lane)) =
        residue
          (NumericPoseidon.permuteState
            (initialNumeric frame assignment) (601 + lane)) := by
      exact congrArg residue interpreted
    _ = semanticLane frame assignment lane := by
      exact congrArg residue
        (permuteState_initial_eq_permute
          frame assignment constantOne lane)

/-! ## Active soundness and inactive gates -/

private theorem satisfies_iff_forall
    (source : List OwnedRow)
    (assignment : ColumnId -> Field) :
    Satisfies source assignment ↔
      ∀ owned, owned ∈ source -> owned.row.Holds assignment := by
  induction source with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [satisfies_cons, inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ owned member
        rcases List.mem_cons.mp member with equal | tailMember
        · subst owned
          exact headHolds
        · exact tailHolds owned tailMember
      · intro all
        exact ⟨
          all head (by simp),
          fun owned member => all owned (by simp [member])
        ⟩

theorem gateRow_active_iff
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (lane : Nat) :
    (gateRow frame lane).row.Holds assignment ↔
      assignment (internalOutputColumn frame lane) =
        assignment (outputColumn frame lane) := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeOne, Fin.one_mul, Fin.mul_one,
    Fin.add_zero, Lean.Grind.Fin.neg_mul]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
      assignment (internalOutputColumn frame lane) -
            assignment (outputColumn frame lane) = 0 ↔
        assignment (internalOutputColumn frame lane) =
          assignment (outputColumn frame lane))

theorem gateRow_complete_of_inactive
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeZero : assignment frame.active = 0)
    (lane : Nat) :
    (gateRow frame lane).row.Holds assignment := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeZero, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul]
  exact Fin.zero_mul _

theorem gateRows_active_output_eq
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (gateRows frame) assignment)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    assignment (internalOutputColumn frame lane) =
      assignment (outputColumn frame lane) := by
  apply (gateRow_active_iff frame assignment activeOne lane).1
  apply (satisfies_iff_forall (gateRows frame) assignment).1 holds
  apply List.mem_map.mpr
  refine ⟨lane, ?_, rfl⟩
  apply List.mem_range.mpr
  simpa [outputWidth, gateRowCount] using laneLt

theorem gateRows_complete_of_inactive
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (activeZero : assignment frame.active = 0) :
    Satisfies (gateRows frame) assignment := by
  apply (satisfies_iff_forall (gateRows frame) assignment).2
  intro owned member
  rcases List.mem_map.mp member with
    ⟨lane, laneMember, equal⟩
  subst owned
  exact gateRow_complete_of_inactive frame assignment activeZero lane

/-- Active satisfaction of the exact 608-row occurrence forces every visible
output lane to equal the executable width-eight production permutation. -/
theorem active_sound
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (rows frame) assignment)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    assignment (outputColumn frame lane) =
      semanticLane frame assignment lane := by
  have split :=
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      assignment).1 holds
  have numericSatisfies :
      Numeric.satisfies NumericPoseidon.rows
        (initialNumeric frame assignment) := by
    exact
      (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
        (columnMap frame) NumericPoseidon.rows assignment).1 split.1
  have numericSound :=
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound
      (fun source => initialNumeric_canonical frame assignment source)
      (initialNumeric_zero frame assignment constantOne)
      numericSatisfies
  have outputKnown :=
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.outputs_known
      (601 + lane) (output_source_mem lane laneLt)
  have executionEqualsInternal :
      execution frame assignment (601 + lane) =
        (assignment (internalOutputColumn frame lane)).val := by
    calc
      execution frame assignment (601 + lane) =
          initialNumeric frame assignment (601 + lane) :=
        numericSound (601 + lane) outputKnown
      _ = (assignment (columnMap frame (601 + lane))).val :=
        rfl
      _ = (assignment (internalOutputColumn frame lane)).val := by
        rw [columnMap_output]
  have internalEqualsSemantic :
      assignment (internalOutputColumn frame lane) =
        semanticLane frame assignment lane := by
    calc
      assignment (internalOutputColumn frame lane) =
          residue
            (assignment (internalOutputColumn frame lane)).val :=
        (residue_field_val
          (assignment (internalOutputColumn frame lane))).symm
      _ = residue (execution frame assignment (601 + lane)) :=
        congrArg residue executionEqualsInternal.symm
      _ = semanticLane frame assignment lane :=
        execution_output_eq_semantic
          frame assignment constantOne lane laneLt
  have gateEquality :=
    gateRows_active_output_eq frame assignment activeOne split.2
      lane laneLt
  exact gateEquality.symm.trans internalEqualsSemantic

/-! ## Honest temporary-only completion -/

/-- Exact ordered values written to the 600 internal SSA allocations. -/
def temporaryValues
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    List Field :=
  (List.range temporaryWidth).map fun index =>
    residue (execution frame assignment (9 + index))

/-- The only honest witness mutation: write the deterministic SSA execution
to this occurrence's exact temporary bundle. -/
def complete
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    ColumnId -> Field :=
  writeColumns assignment frame.temporaries.ids
    (temporaryValues frame assignment)

@[simp] theorem temporaryValues_length
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    (temporaryValues frame assignment).length = temporaryWidth := by
  simp [temporaryValues]

theorem Frame.temporary_ids_nodup (frame : Frame) :
    frame.temporaries.ids.Nodup := by
  have split := frame.allocationsNodup
  rw [List.nodup_append] at split
  exact split.2.1

theorem complete_changesOnly
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    ChangesOnly frame.temporaries.ids assignment
      (complete frame assignment) := by
  exact writeColumns_changesOnly assignment frame.temporaries.ids
    (temporaryValues frame assignment)

theorem complete_agrees_visible
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
      (complete frame assignment) := by
  exact writeColumns_agreesOn assignment frame.temporaries.ids
    frame.visibleIds (temporaryValues frame assignment)
    frame.temporariesDisjointVisible

private theorem temporaryValues_getD
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < temporaryWidth)
    (fallback : Field) :
    (temporaryValues frame assignment).getD index fallback =
      residue (execution frame assignment (9 + index)) := by
  have valuesLt :
      index < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    exact indexLt
  rw [← List.getElem_eq_getD
    (l := temporaryValues frame assignment)
    (i := index) (h := valuesLt) fallback]
  simp [temporaryValues]

theorem complete_temporary
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < temporaryWidth) :
    complete frame assignment (temporaryColumn frame index) =
      residue (execution frame assignment (9 + index)) := by
  have recovered :
      frame.temporaries.ids.map (complete frame assignment) =
        temporaryValues frame assignment := by
    apply writeColumns_map_eq
    · rw [Frame.temporary_ids_length, temporaryValues_length]
    · exact frame.temporary_ids_nodup
  have atIndex := congrArg
    (fun values : List Field =>
      values.getD index (complete frame assignment frame.one))
    recovered
  have idsLt : index < frame.temporaries.ids.length := by
    rw [Frame.temporary_ids_length]
    exact indexLt
  have mappedIdsLt :
      index <
        (frame.temporaries.ids.map
          (complete frame assignment)).length := by
    simpa using idsLt
  have valuesLt :
      index < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    exact indexLt
  change
    (frame.temporaries.ids.map (complete frame assignment)).getD
          index (complete frame assignment frame.one) =
        (temporaryValues frame assignment).getD
          index (complete frame assignment frame.one)
    at atIndex
  rw [← List.getElem_eq_getD
      (l := frame.temporaries.ids.map (complete frame assignment))
      (i := index) (h := mappedIdsLt)
      (complete frame assignment frame.one),
    ← List.getElem_eq_getD
      (l := temporaryValues frame assignment)
      (i := index) (h := valuesLt)
      (complete frame assignment frame.one)] at atIndex
  simp only [List.getElem_map] at atIndex
  rw [List.getElem_eq_getD
      (l := frame.temporaries.ids) (i := index)
      (h := idsLt) frame.one,
    List.getElem_eq_getD
      (l := temporaryValues frame assignment) (i := index)
      (h := valuesLt) (complete frame assignment frame.one)]
    at atIndex
  rw [temporaryValues_getD frame assignment index indexLt] at atIndex
  exact atIndex

theorem execution_canonical
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (source : Nat) :
    execution frame assignment source < Numeric.modulus := by
  exact
    Nightstream.Implementation.R1CS.Program.run_canonical
      (fun column =>
        initialNumeric_canonical frame assignment column)
      source

private theorem inputColumn_mem
    (frame : Frame)
    (lane : Nat)
    (laneLt : lane < inputWidth) :
    inputColumn frame lane ∈ frame.input.ids := by
  have idsLt : lane < frame.input.ids.length := by
    rw [Frame.input_ids_length]
    exact laneLt
  unfold inputColumn
  rw [← List.getElem_eq_getD
    (l := frame.input.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem outputColumn_mem
    (frame : Frame)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    outputColumn frame lane ∈ frame.output.ids := by
  have idsLt : lane < frame.output.ids.length := by
    rw [Frame.output_ids_length]
    exact laneLt
  unfold outputColumn
  rw [← List.getElem_eq_getD
    (l := frame.output.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem columnMap_mem_visible_of_lt
    (frame : Frame)
    (source : Nat)
    (sourceLt : source < 9) :
    columnMap frame source ∈ frame.visibleIds := by
  by_cases sourceZero : source = 0
  · subst source
    simp [Frame.visibleIds, columnMap_zero]
  · have sourcePositive : 0 < source :=
      Nat.pos_of_ne_zero sourceZero
    have laneLt : source - 1 < inputWidth := by
      unfold inputWidth
      omega
    have sourceEq : source - 1 + 1 = source := by
      omega
    have mapped :
        columnMap frame source =
          inputColumn frame (source - 1) := by
      calc
        columnMap frame source =
            columnMap frame (source - 1 + 1) := by rw [sourceEq]
        _ = inputColumn frame (source - 1) :=
          columnMap_input frame (source - 1) laneLt
    rw [mapped]
    simp [Frame.visibleIds,
      inputColumn_mem frame (source - 1) laneLt]

private theorem columnMap_temporary
    (frame : Frame)
    (source : Nat)
    (sourceGe : 9 ≤ source) :
    columnMap frame source =
      temporaryColumn frame (source - 9) := by
  unfold columnMap
  rw [if_neg (by omega), if_neg (by omega)]

theorem residue_val_of_lt
    (value : Nat)
    (valueLt : value < Numeric.modulus) :
    (residue value).val = value := by
  change
    value % Nightstream.SuperNeo.Concrete.goldilocksModulus = value
  apply Nat.mod_eq_of_lt
  simpa [Numeric.modulus,
    Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus] using valueLt

private theorem run_eq_of_not_output
    (state : Nat -> Nat)
    (definitions :
      List Nightstream.Implementation.R1CS.Program.Definition)
    (source : Nat)
    (notOutput :
      ∀ definition, definition ∈ definitions ->
        definition.output ≠ source) :
    Nightstream.Implementation.R1CS.Program.run
        state definitions source =
      state source := by
  induction definitions generalizing state with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headDifferent : head.output ≠ source :=
        notOutput head (by simp)
      have tailDifferent :
          ∀ definition, definition ∈ tail ->
            definition.output ≠ source := by
        intro definition member
        exact notOutput definition (by simp [member])
      calc
        Nightstream.Implementation.R1CS.Program.run
              state (head :: tail) source =
            Nightstream.Implementation.R1CS.Program.run
              (Nightstream.Implementation.R1CS.Program.execute state head)
              tail source :=
          rfl
        _ =
            Nightstream.Implementation.R1CS.Program.execute
              state head source :=
          inductionHypothesis
            (Nightstream.Implementation.R1CS.Program.execute state head)
            tailDifferent
        _ = state source := by
          unfold Nightstream.Implementation.R1CS.Program.execute
          exact
            Nightstream.Implementation.R1CS.Program.setColumn_other
              state headDifferent.symm

private theorem definition_outputs_lt_columnCount :
    ∀ definition ∈ NumericPoseidon.definitions,
      definition.output < 609 := by
  decide

private theorem columnMap_eq_one_of_columnCount_le
    (frame : Frame)
    (source : Nat)
    (sourceGe : 609 ≤ source) :
    columnMap frame source = frame.one := by
  unfold columnMap temporaryColumn
  rw [if_neg (by omega), if_neg (by omega)]
  have outside :
      frame.temporaries.ids.length ≤ source - 9 := by
    rw [Frame.temporary_ids_length]
    unfold temporaryWidth
    omega
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_none outside]
  rfl

/-- The completed typed assignment, pulled back through the exact occurrence
map, is definitionally the deterministic numeric SSA execution. -/
theorem completedNumeric_eq_execution
    (frame : Frame)
    (assignment : ColumnId -> Field) :
    numericAssignment (columnMap frame) (complete frame assignment) =
      execution frame assignment := by
  funext source
  by_cases sourceLtNine : source < 9
  · have preserved :=
      complete_agrees_visible frame assignment
        (columnMap frame source)
        (columnMap_mem_visible_of_lt frame source sourceLtNine)
    have runPreserved :=
      Nightstream.Implementation.R1CS.Program.run_preserves_known
        Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions_wellFormed
        (initialNumeric frame assignment)
        source (input_source_mem source sourceLtNine)
    calc
      numericAssignment (columnMap frame)
            (complete frame assignment) source =
          (complete frame assignment (columnMap frame source)).val :=
        rfl
      _ = (assignment (columnMap frame source)).val :=
        congrArg Fin.val preserved
      _ = initialNumeric frame assignment source :=
        rfl
      _ = execution frame assignment source :=
        runPreserved.symm
  · by_cases sourceLtColumnCount : source < 609
    · have sourceGe : 9 ≤ source := Nat.le_of_not_gt sourceLtNine
      have indexLt : source - 9 < temporaryWidth := by
        unfold temporaryWidth
        omega
      calc
        numericAssignment (columnMap frame)
              (complete frame assignment) source =
            (complete frame assignment
              (temporaryColumn frame (source - 9))).val := by
          rw [numericAssignment,
            columnMap_temporary frame source sourceGe]
        _ =
            (residue
              (execution frame assignment
                (9 + (source - 9)))).val := by
          rw [complete_temporary frame assignment
            (source - 9) indexLt]
        _ = execution frame assignment source := by
          rw [show 9 + (source - 9) = source by omega]
          exact residue_val_of_lt _
            (execution_canonical frame assignment source)
    · have sourceGe : 609 ≤ source :=
        Nat.le_of_not_gt sourceLtColumnCount
      have mapped :=
        columnMap_eq_one_of_columnCount_le frame source sourceGe
      have oneVisible : frame.one ∈ frame.visibleIds := by
        simp [Frame.visibleIds]
      have onePreserved :=
        complete_agrees_visible frame assignment frame.one oneVisible
      have runPreserved :
          execution frame assignment source =
            initialNumeric frame assignment source := by
        apply run_eq_of_not_output
        intro definition member
        have definitionLt :=
          definition_outputs_lt_columnCount definition member
        omega
      calc
        numericAssignment (columnMap frame)
              (complete frame assignment) source =
            (complete frame assignment frame.one).val := by
          rw [numericAssignment, mapped]
        _ = (assignment frame.one).val :=
          congrArg Fin.val onePreserved
        _ = initialNumeric frame assignment source := by
          rw [initialNumeric, numericAssignment, mapped]
        _ = execution frame assignment source :=
          runPreserved.symm

theorem core_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1) :
    Satisfies (coreRows frame) (complete frame assignment) := by
  apply
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) NumericPoseidon.rows
      (complete frame assignment)).2
  rw [completedNumeric_eq_execution]
  exact
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete
      (fun source =>
        initialNumeric_canonical frame assignment source)
      (initialNumeric_zero frame assignment constantOne)

theorem gateRows_complete_of_active
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane, lane < outputWidth ->
        assignment (outputColumn frame lane) =
          semanticLane frame assignment lane) :
    Satisfies (gateRows frame) (complete frame assignment) := by
  apply
    (satisfies_iff_forall (gateRows frame)
      (complete frame assignment)).2
  intro owned member
  rcases List.mem_map.mp member with
    ⟨lane, laneMember, equal⟩
  subst owned
  have laneLtGate : lane < gateRowCount :=
    List.mem_range.mp laneMember
  have laneLt : lane < outputWidth := by
    simpa [gateRowCount, outputWidth] using laneLtGate
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      complete frame assignment frame.active = 1 := by
    exact
      (complete_agrees_visible frame assignment
        frame.active activeVisible).trans activeOne
  apply
    (gateRow_active_iff frame (complete frame assignment)
      completedActive lane).2
  have internalIndexLt : 592 + lane < temporaryWidth := by
    unfold temporaryWidth outputWidth at *
    omega
  have outputVisible :
      outputColumn frame lane ∈ frame.visibleIds := by
    simp [Frame.visibleIds, outputColumn_mem frame lane laneLt]
  have outputPreserved :=
    complete_agrees_visible frame assignment
      (outputColumn frame lane) outputVisible
  calc
    complete frame assignment (internalOutputColumn frame lane) =
        residue
          (execution frame assignment (9 + (592 + lane))) := by
      exact complete_temporary frame assignment
        (592 + lane) internalIndexLt
    _ = residue (execution frame assignment (601 + lane)) := by
      congr 2
      omega
    _ = semanticLane frame assignment lane :=
      execution_output_eq_semantic
        frame assignment constantOne lane laneLt
    _ = assignment (outputColumn frame lane) :=
      (outputsCorrect lane laneLt).symm
    _ = complete frame assignment (outputColumn frame lane) :=
      outputPreserved.symm

/-- Honest active inputs extend by writing only the 600 receipt-owned
temporaries. No visible input or output is rewritten. -/
theorem active_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane, lane < outputWidth ->
        assignment (outputColumn frame lane) =
          semanticLane frame assignment lane) :
    Satisfies (rows frame) (complete frame assignment) := by
  apply
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (complete frame assignment)).2
  exact ⟨
    core_complete frame assignment constantOne,
    gateRows_complete_of_active frame assignment constantOne
      activeOne outputsCorrect
  ⟩

/-- Inactive occurrences still execute and satisfy the internal permutation
rows, while all eight visible-output copies are vacuous. -/
theorem inactive_complete
    (frame : Frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0) :
    Satisfies (rows frame) (complete frame assignment) := by
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      complete frame assignment frame.active = 0 := by
    exact
      (complete_agrees_visible frame assignment
        frame.active activeVisible).trans activeZero
  apply
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (complete frame assignment)).2
  exact ⟨
    core_complete frame assignment constantOne,
    gateRows_complete_of_inactive frame
      (complete frame assignment) completedActive
  ⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe
