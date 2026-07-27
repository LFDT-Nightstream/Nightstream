import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients

/-!
Contract: activation-aware typed occurrence of the selected fixed-23
Poseidon2 sponge core.

Owns: the exact map from the canonical numeric program to typed input,
temporary, and output bundles; the four activation-gated digest copies; and
the nonoptional physical receipt.

Does not own: the optional-digest/alignment wrapper of `hashPrior` or
`hashNext`, their typed preimage codecs, or collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Canonical

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership
open Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants

abbrev rows : List Nightstream.Implementation.R1CS.Row :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program selected

theorem rows_length : rows.length = 2464 :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program_length
    selected

end Canonical

def inputWidth : Nat := 23
def outputWidth : Nat := 4
def temporaryWidth : Nat := 2464
def coreRowCount : Nat := 2464
def gateRowCount : Nat := 4
def recurringRows : Nat := coreRowCount + gateRowCount

def inputLayout : Layout := auxiliaryLayout inputWidth
def outputLayout : Layout := auxiliaryLayout outputWidth
def temporaryLayout : Layout := auxiliaryLayout temporaryWidth

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
  change (List.replicate temporaryWidth Ownership.auxiliaryColumn).length =
    temporaryWidth
  exact List.length_replicate ..

theorem temporary_ids_nodup (frame : Frame) :
    frame.temporaries.ids.Nodup := by
  have split := frame.allocationsNodup
  rw [List.nodup_append] at split
  exact split.2.1

end Frame

def inputColumn (frame : Frame) (index : Nat) : ColumnId :=
  frame.input.ids.getD index frame.one

def outputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  frame.output.ids.getD lane frame.one

def temporaryColumn (frame : Frame) (index : Nat) : ColumnId :=
  frame.temporaries.ids.getD index frame.one

def sourceTemporaryIndex (source : Nat) : Nat :=
  let call := source / 361
  let offset := source % 361
  if offset < 17 then
    call * 352 + 344 + (offset - 9)
  else
    call * 352 + (offset - 17)

/-- Source column zero is constant one, `2527..2549` are the ordered inputs,
and the 2,464 receipt-owned source columns map positionally to the typed
temporary bundle. Other source columns are provably absent from emitted rows. -/
def columnMap (frame : Frame) (source : Nat) : ColumnId :=
  if source = 0 then frame.one
  else if 2527 ≤ source ∧ source < 2550 then
    inputColumn frame (source - 2527)
  else if source < 2527 then
    temporaryColumn frame (sourceTemporaryIndex source)
  else frame.one

@[simp] theorem columnMap_zero (frame : Frame) :
    columnMap frame 0 = frame.one := by
  simp [columnMap]

theorem columnMap_input
    (frame : Frame) (index : Nat) (indexLt : index < inputWidth) :
    columnMap frame (2527 + index) = inputColumn frame index := by
  unfold inputWidth at indexLt
  unfold columnMap
  rw [if_neg (by omega), if_pos (by omega)]
  congr 1
  omega

private theorem sourceTemporary_call_lt
    (position :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.TemporaryPosition) :
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
        position < 7 :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall_lt
    position

private theorem sourceTemporary_within_lt
    (position :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.TemporaryPosition) :
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryWithinCall
        position < 352 := by
  simpa [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.perCallTemporaries]
    using
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryWithinCall_lt
        position

theorem sourceTemporaryIndex_sourceColumn
    (position :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.TemporaryPosition) :
    sourceTemporaryIndex
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position) =
      position.val := by
  have callLt := sourceTemporary_call_lt position
  have withinLt := sourceTemporary_within_lt position
  have offsetLt :=
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset_lt
      position
  have offsetLt361 :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
          position < 361 := by
    simpa only [
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.callStride_eq
    ] using offsetLt
  have positionEq :
      position.val =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
            position * 352 +
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryWithinCall
            position := by
    unfold
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryWithinCall
    rw [Nat.mul_comm]
    exact (Nat.div_add_mod position.val 352).symm
  rw [show
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
            position * 361 +
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
            position by
      simp only [
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.callStride_eq]]
  unfold sourceTemporaryIndex
  rw [show
      (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
          position * 361 +
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
          position) / 361 =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
          position by
      rw [Nat.mul_comm, Nat.mul_add_div (by decide : 0 < 361),
        Nat.div_eq_of_lt offsetLt361, Nat.add_zero],
    show
      (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryCall
          position * 361 +
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
          position) % 361 =
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
          position by
      exact Nat.mul_add_mod_of_lt offsetLt361]
  unfold
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
  simp only [
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.sboxTemporaries
  ] at withinLt positionEq ⊢
  by_cases first : 
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryWithinCall
          position < 344
  · rw [if_pos first, if_neg (by omega)]
    omega
  · rw [if_neg first, if_pos (by omega)]
    omega

theorem columnMap_sourceTemporary
    (frame : Frame)
    (position :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.TemporaryPosition) :
    columnMap frame
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position) =
      temporaryColumn frame position.val := by
  have below :=
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn_lt_inputBase
      position
  have offsetGe :
      9 ≤
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
          position := by
    unfold
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryOffset
    have withinLt := sourceTemporary_within_lt position
    split <;> omega
  have nonzero :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position ≠ 0 := by
    unfold
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
    omega
  unfold columnMap
  rw [if_neg nonzero,
    if_neg (by
      rw [Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq]
        at below
      omega),
    if_pos (by
      simpa [
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq
      ] using below),
    sourceTemporaryIndex_sourceColumn]

def coreRows (frame : Frame) : List OwnedRow :=
  ownedRowsFrom frame.owner frame.firstOrdinal (columnMap frame)
    Canonical.rows

def internalOutputColumn (frame : Frame) (lane : Nat) : ColumnId :=
  columnMap frame
    ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
      ⟨lane %
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width,
        Nat.mod_lt _ (by decide)⟩)

def gateRow (frame : Frame) (lane : Nat) : OwnedRow where
  id := {
    owner := frame.owner
    ordinal := frame.firstOrdinal + coreRowCount + lane
  }
  row := {
    a := singleton frame.active 1
    b := difference (internalOutputColumn frame lane) (outputColumn frame lane)
    c := []
  }

def gateRows (frame : Frame) : List OwnedRow :=
  (List.range gateRowCount).map (gateRow frame)

def rows (frame : Frame) : List OwnedRow :=
  coreRows frame ++ gateRows frame

def footprint : CallFootprint where
  recurringRows := recurringRows
  temporaries := [temporaryLayout]

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
  rw [coreRows, ownedRowsFrom_length, Canonical.rows_length]
  rfl

theorem gateRows_length (frame : Frame) :
    (gateRows frame).length = gateRowCount := by
  simp [gateRows]

theorem rows_length (frame : Frame) :
    (rows frame).length = recurringRows := by
  rw [rows, List.length_append, coreRows_length, gateRows_length]
  rfl

theorem receipt_row_count (frame : Frame) :
    (receipt frame).rows.length = footprint.recurringRows :=
  rows_length frame

theorem rows_owned
    (frame : Frame) (owned : OwnedRow) (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_append.mp member with core | gate
  · exact ownedRowsFrom_owned frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows owned core
  · rcases List.mem_map.mp gate with ⟨lane, _, rfl⟩
    rfl

private theorem gateIds_nodup_of
    (frame : Frame) (lanes : List Nat) (nodup : lanes.Nodup) :
    ((lanes.map (gateRow frame)).map (fun row => row.id)).Nodup := by
  rw [List.map_map]
  change (lanes.map (fun lane => (gateRow frame lane).id)).Nodup
  induction lanes with
  | nil => exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have split : head ∉ tail ∧ tail.Nodup := by
        simpa only [List.nodup_cons] using nodup
      rw [List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.mp member with ⟨lane, laneMember, equal⟩
        have ordinalEqual := congrArg RowId.ordinal equal
        simp only [gateRow] at ordinalEqual
        have laneEqual : lane = head := by omega
        exact split.1 (laneEqual ▸ laneMember)
      · exact inductionHypothesis split.2

private theorem gateRows_ids_nodup (frame : Frame) :
    ((gateRows frame).map (fun row => row.id)).Nodup :=
  gateIds_nodup_of frame (List.range gateRowCount) List.nodup_range

private theorem coreRow_ordinal_lt
    (frame : Frame) (owned : OwnedRow) (member : owned ∈ coreRows frame) :
    owned.id.ordinal < frame.firstOrdinal + coreRowCount := by
  have mappedMember :
      owned.id ∈ (coreRows frame).map (fun row => row.id) :=
    List.mem_map.mpr ⟨owned, member, rfl⟩
  rw [coreRows,
    ownedRowsFrom_ids_exact frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows] at mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨ordinal, ordinalMember, equal⟩
  rcases List.mem_range'.mp ordinalMember with
    ⟨offset, offsetLt, ordinalEqual⟩
  have exactOrdinal : ordinal = owned.id.ordinal := by
    simpa using congrArg RowId.ordinal equal
  have ordinalLt :
      ordinal < frame.firstOrdinal + Canonical.rows.length := by
    simp only [Nat.one_mul] at ordinalEqual
    omega
  calc
    owned.id.ordinal = ordinal := exactOrdinal.symm
    _ < frame.firstOrdinal + Canonical.rows.length := ordinalLt
    _ = frame.firstOrdinal + coreRowCount := by
      rw [Canonical.rows_length]
      rfl

private theorem gateRow_ordinal_ge
    (frame : Frame) (owned : OwnedRow) (member : owned ∈ gateRows frame) :
    frame.firstOrdinal + coreRowCount ≤ owned.id.ordinal := by
  rcases List.mem_map.mp member with ⟨lane, _, rfl⟩
  simp [gateRow]

theorem rowIds_nodup (frame : Frame) :
    ((rows frame).map (fun row => row.id)).Nodup := by
  rw [rows, List.map_append, List.nodup_append]
  refine ⟨
    ownedRowsFrom_ids_nodup frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows,
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

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe
