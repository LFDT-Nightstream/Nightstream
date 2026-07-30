import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace

/-!
Contract: give one canonical public-PiRLC quotient occurrence a complete typed
physical placement.

Owns:
- a stable translation of numeric source columns into typed `ColumnId`s;
- one contiguous, duplicate-free auxiliary allocation owned by the occurrence;
- exact contiguous row identities and row ownership;
- exact numeric/typed satisfaction transport; and
- complete row support by the constant wire, pre-existing visible reads, or
  the declared auxiliary block.

Does not own: the surrounding `nifsVerify` call frame, the transcript that
derives `beta`, semantic decoding of public NIFS inputs, the remaining
PiCCS/PiDEC verifier rows, or the final call recipe.

The `Columns.BelowBase` premise is load-bearing.  It rules out aliasing between
authoritative public columns and the fresh quotient-program witness block.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace
open Nightstream.Implementation.R1CS.Canonical.KTraceProgram

/-- Exact auxiliary width of a batch, derived from its trace list. -/
def batchAuxWidth (layout : BatchLayout) : Nat :=
  (layout.traces.map traceAuxWidth).sum

/-- One owner-local auxiliary coordinate. -/
def auxiliaryId (owner : PhysicalOwner) (offset : Nat) : ColumnId where
  owner := owner
  bundleIndex := 1
  coordinateIndex := offset

/-- One pre-existing numeric source coordinate.

Bundle zero is reserved for reads; bundle one is reserved for this
occurrence's fresh auxiliaries.  Consequently reads and writes stay distinct
even when the surrounding program chooses the same structural owner for both.
-/
def sourceId (owner : PhysicalOwner) (source : Nat) : ColumnId where
  owner := owner
  bundleIndex := 0
  coordinateIndex := source

/-- Stable numeric-to-typed placement.  Numeric column zero is the global
constant wire, the selected auxiliary interval is occurrence-owned, and every
other coordinate is a pre-existing source read. -/
def columnMap
    (sourceOwner occurrenceOwner : PhysicalOwner)
    (base width source : Nat) : ColumnId :=
  if source = 0 then
    { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 }
  else if base ≤ source ∧ source < base + width then
    auxiliaryId occurrenceOwner (source - base)
  else
    sourceId sourceOwner source

/-- The exact fresh auxiliary list. -/
def auxiliaryColumns
    (owner : PhysicalOwner) (width : Nat) : List OwnedColumn :=
  (List.range width).map fun offset =>
    { id := auxiliaryId owner offset
      ownership := .auxiliaryColumn }

def numericColumns (row : Numeric.Row) : List Nat :=
  (row.a ++ row.b ++ row.c).map Prod.fst

def numericSupport (rows : List Numeric.Row) : List Nat :=
  rows.flatMap numericColumns

/-- Pre-existing reads, excluding the global constant and the fresh auxiliary
interval.  The list deliberately retains occurrence order and duplicates:
ownership is attached to allocated columns, while support is a dependency
list. -/
def visibleSourceColumns (base width : Nat) (rows : List Numeric.Row) :
    List Nat :=
  (numericSupport rows).filter fun source =>
    decide (source ≠ 0 ∧ ¬ (base ≤ source ∧ source < base + width))

def visibleIds
    (sourceOwner : PhysicalOwner) (base width : Nat)
    (rows : List Numeric.Row) : List ColumnId :=
  (visibleSourceColumns base width rows).map (sourceId sourceOwner)

/-- A physical occurrence is constructed from a concrete public-PiRLC batch.
The only caller choices are structural owners and the first row ordinal. -/
structure PhysicalOccurrence
    {arity matrixCount : Nat}
    (base : Nat)
    (columns : Columns arity matrixCount)
    (valid : columns.Valid) where
  sourceOwner : PhysicalOwner
  owner : PhysicalOwner
  firstOrdinal : Nat
  basePositive : 0 < base
  sourcesBelowBase : columns.BelowBase base

def PhysicalOccurrence.numeric
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (_physical : PhysicalOccurrence base columns valid) :
    Nightstream.Implementation.R1CS.Canonical.KTraceProgram.Occurrence :=
  Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.occurrence
    base columns valid

def PhysicalOccurrence.auxWidth
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) : Nat :=
  batchAuxWidth physical.numeric.layout

def PhysicalOccurrence.map
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) : Nat -> ColumnId :=
  columnMap physical.sourceOwner physical.owner base physical.auxWidth

def PhysicalOccurrence.rows
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) : List OwnedRow :=
  ownedRowsFrom physical.owner physical.firstOrdinal physical.map
    physical.numeric.rows

def PhysicalOccurrence.auxiliaries
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) : List OwnedColumn :=
  auxiliaryColumns physical.owner physical.auxWidth

def PhysicalOccurrence.visible
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) : List ColumnId :=
  { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 } ::
    visibleIds physical.sourceOwner base physical.auxWidth physical.numeric.rows

@[simp] theorem auxiliaryColumns_length
    (owner : PhysicalOwner) (width : Nat) :
    (auxiliaryColumns owner width).length = width := by
  simp [auxiliaryColumns]

theorem auxiliaryColumns_ids_nodup
    (owner : PhysicalOwner) (width : Nat) :
    ((auxiliaryColumns owner width).map (fun column => column.id)).Nodup := by
  simp only [auxiliaryColumns, List.map_map]
  exact List.nodup_range.map
    (fun offset => auxiliaryId owner offset) (by
      intro left right different equal
      apply different
      exact congrArg ColumnId.coordinateIndex equal)

theorem rows_length
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) :
    physical.rows.length =
      (23 + 2 * matrixCount) * (321 * arity + 482) := by
  rw [PhysicalOccurrence.rows, ownedRowsFrom_length]
  simpa only [PhysicalOccurrence.numeric] using
    occurrence_rows_length base columns valid

theorem rows_owned
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (row : OwnedRow) (member : row ∈ physical.rows) :
    row.id.owner = physical.owner :=
  ownedRowsFrom_owned physical.owner physical.firstOrdinal physical.map
    physical.numeric.rows row member

theorem row_ids_nodup
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid) :
    (physical.rows.map fun row => row.id).Nodup :=
  ownedRowsFrom_ids_nodup physical.owner physical.firstOrdinal physical.map
    physical.numeric.rows

theorem auxiliary_owned
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (column : OwnedColumn) (member : column ∈ physical.auxiliaries) :
    column.id.owner = physical.owner := by
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  rfl

/-- Every translated row dependency is accounted exactly once as the constant
wire, a pre-existing read, or one of the occurrence's auxiliary columns. -/
theorem rows_supported
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (owned : OwnedRow) (ownedMember : owned ∈ physical.rows)
    (column : ColumnId) (columnMember : column ∈ owned.columnIds) :
    column ∈ physical.visible ++
      physical.auxiliaries.map (fun allocated => allocated.id) := by
  unfold PhysicalOccurrence.rows at ownedMember
  have mappedMember :
      owned.row ∈
        (ownedRowsFrom physical.owner physical.firstOrdinal physical.map
          physical.numeric.rows).map (fun row => row.row) :=
    List.mem_map.mpr ⟨owned, ownedMember, rfl⟩
  rw [ownedRowsFrom_rows] at mappedMember
  rcases List.mem_map.mp mappedMember with
    ⟨sourceRow, sourceRowMember, rowEqual⟩
  change column ∈ owned.row.columnIds at columnMember
  rw [← rowEqual, NumericRowBridge.row_columnIds] at columnMember
  rcases List.mem_map.mp columnMember with ⟨source, sourceMember, rfl⟩
  have inSupport : source.1 ∈ numericSupport physical.numeric.rows := by
    unfold numericSupport
    apply List.mem_flatMap.mpr
    exact ⟨sourceRow, sourceRowMember, by
      unfold numericColumns
      exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩⟩
  by_cases zero : source.1 = 0
  · rw [PhysicalOccurrence.map, columnMap, if_pos zero]
    exact List.mem_append_left _ (List.mem_cons_self)
  by_cases allocated :
      base ≤ source.1 ∧ source.1 < base + physical.auxWidth
  · rw [PhysicalOccurrence.map, columnMap, if_neg zero, if_pos allocated]
    apply List.mem_append_right
    unfold PhysicalOccurrence.auxiliaries auxiliaryColumns
    simp only [List.map_map]
    apply List.mem_map.mpr
    refine ⟨source.1 - base, List.mem_range.mpr ?_, rfl⟩
    omega
  · rw [PhysicalOccurrence.map, columnMap, if_neg zero, if_neg allocated]
    apply List.mem_append_left
    simp only [PhysicalOccurrence.visible, List.mem_cons]
    right
    unfold visibleIds visibleSourceColumns
    apply List.mem_map.mpr
    refine ⟨source.1, ?_, rfl⟩
    apply List.mem_filter.mpr
    exact ⟨inSupport, by simp [zero, allocated]⟩

/-- Typed satisfaction is exactly numeric satisfaction on the canonical
representatives of the same physical columns. -/
theorem satisfies_iff
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (assignment : ColumnId -> Nightstream.SuperNeo.Concrete.F) :
    Satisfies physical.rows assignment ↔
      Nightstream.Implementation.R1CS.Satisfies physical.numeric.rows
        (numericAssignment physical.map assignment) :=
  ownedRowsFrom_satisfies_iff physical.owner physical.firstOrdinal
    physical.map physical.numeric.rows assignment

private def constantId : ColumnId where
  owner := .prelude
  bundleIndex := 0
  coordinateIndex := 0

/-- Lift a numeric witness into the occurrence's exact typed column space.
Source columns keep their pre-existing owner, while fresh numeric coordinates
in the selected auxiliary interval are placed in the occurrence-owned bundle.
All values are reduced to their canonical Goldilocks representatives. -/
def PhysicalOccurrence.liftAssignment
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (witness : Nat → Nat) :
    ColumnId → Nightstream.SuperNeo.Concrete.F :=
  fun column =>
    if column = constantId then
      residue (witness 0)
    else if
        column.owner = physical.owner ∧
          column.bundleIndex = 1 ∧
          column.coordinateIndex < physical.auxWidth then
      residue (witness (base + column.coordinateIndex))
    else if
        column.owner = physical.sourceOwner ∧
          column.bundleIndex = 0 then
      residue (witness column.coordinateIndex)
    else
      0

theorem PhysicalOccurrence.liftAssignment_map
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (witness : Nat → Nat) (source : Nat) :
    physical.liftAssignment witness (physical.map source) =
      residue (witness source) := by
  by_cases zero : source = 0
  · subst source
    simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, constantId]
  by_cases allocated :
      base ≤ source ∧ source < base + physical.auxWidth
  · have offsetLow : source - base < physical.auxWidth := by omega
    have recover : base + (source - base) = source := by omega
    simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, zero, allocated, constantId, auxiliaryId, offsetLow,
      recover]
  · simp [PhysicalOccurrence.liftAssignment, PhysicalOccurrence.map,
      columnMap, zero, allocated, constantId, sourceId]

theorem PhysicalOccurrence.numericAssignment_liftAssignment
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (witness : Nat → Nat) :
    numericAssignment physical.map (physical.liftAssignment witness) =
      canonicalAssignment witness := by
  funext source
  unfold numericAssignment canonicalAssignment
  rw [physical.liftAssignment_map]
  rfl

/-- Typed honest completeness for the physically placed public-PiRLC
occurrence.  Frozen coefficient exactness constructs the numeric auxiliary
witness, and this theorem places that witness in the declared typed columns.
No row equation or acceptance proposition is supplied by the caller. -/
theorem PhysicalOccurrence.rows_honest
    {arity matrixCount base}
    {columns : Columns arity matrixCount}
    {valid : columns.Valid}
    (physical : PhysicalOccurrence base columns valid)
    (source : Nat → Nat)
    (constantWire : source 0 = 1)
    (exact : physical.numeric.Exact source) :
    ∃ assignment : ColumnId → Nightstream.SuperNeo.Concrete.F,
      Satisfies physical.rows assignment := by
  let witness :=
    Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.honestWitness
      source base columns
  have numericSatisfied :
      Nightstream.Implementation.R1CS.Satisfies
        physical.numeric.rows witness :=
    Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace.occurrence_rows_honest
      source base columns valid
      physical.basePositive physical.sourcesBelowBase constantWire exact
  have canonicalSatisfied :
      Nightstream.Implementation.R1CS.Satisfies
        physical.numeric.rows (canonicalAssignment witness) :=
    satisfies_canonical physical.numeric.rows witness numericSatisfied
  refine ⟨physical.liftAssignment witness, ?_⟩
  apply (satisfies_iff physical (physical.liftAssignment witness)).2
  rw [physical.numericAssignment_liftAssignment]
  exact canonicalSatisfied

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KPiRlcPhysicalOccurrence
