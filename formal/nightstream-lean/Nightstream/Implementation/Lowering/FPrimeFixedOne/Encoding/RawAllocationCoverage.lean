import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.AllocationCoverage

/-!
Contract: the small converse to row conservation used by the selected NIFS
assembly.

Conservation proves that emitted rows do not escape a declared allocation.
`NumericRowsCover` and `TypedRowsCover` prove the other direction: every
declared auxiliary is mentioned by an emitted row.  The pair rejects a
declared-but-unwritten column even when all row and allocation counts happen to
agree.

This module owns only the generic predicates and composition/translation
lemmas.  Concrete recipes must prove coverage from their own row and allocation
definitions.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.Canonical

private abbrev NumericRow := Nightstream.Implementation.R1CS.Row
private abbrev TypedRow :=
  Nightstream.Implementation.Lowering.Goldilocks.Row

/-- Ordered numeric support of one row. -/
def numericColumnIds (row : NumericRow) : List Nat :=
  (row.a ++ row.b ++ row.c).map Prod.fst

/-- Every declared numeric column is reached by an emitted row. -/
def NumericRowsCover
    (rows : List NumericRow) (columns : List Nat) : Prop :=
  ∀ column, column ∈ columns →
    ∃ row ∈ rows, column ∈ numericColumnIds row

/-- Every declared typed column is reached by an emitted row. -/
def TypedRowsCover
    (rows : List TypedRow) (columns : List ColumnId) : Prop :=
  ∀ column, column ∈ columns →
    ∃ row ∈ rows, column ∈ row.columnIds

theorem numeric_append
    (leftRows rightRows : List NumericRow)
    (leftColumns rightColumns : List Nat)
    (left : NumericRowsCover leftRows leftColumns)
    (right : NumericRowsCover rightRows rightColumns) :
    NumericRowsCover (leftRows ++ rightRows)
      (leftColumns ++ rightColumns) := by
  intro column member
  rcases List.mem_append.1 member with inLeft | inRight
  · rcases left column inLeft with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · rcases right column inRight with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_right _ rowMember, mentioned⟩

theorem typed_append
    (leftRows rightRows : List TypedRow)
    (leftColumns rightColumns : List ColumnId)
    (left : TypedRowsCover leftRows leftColumns)
    (right : TypedRowsCover rightRows rightColumns) :
    TypedRowsCover (leftRows ++ rightRows)
      (leftColumns ++ rightColumns) := by
  intro column member
  rcases List.mem_append.1 member with inLeft | inRight
  · rcases left column inLeft with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · rcases right column inRight with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_right _ rowMember, mentioned⟩

/-- Numeric coverage survives the selected physical column translation. -/
theorem translate
    (columnMap : Nat → ColumnId)
    (rows : List NumericRow) (columns : List Nat)
    (covered : NumericRowsCover rows columns) :
    TypedRowsCover
      (rows.map
        (Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row
          columnMap))
      (columns.map columnMap) := by
  intro column member
  rcases List.mem_map.1 member with ⟨sourceColumn, sourceMember, rfl⟩
  rcases covered sourceColumn sourceMember with
    ⟨sourceRow, rowMember, mentioned⟩
  refine
    ⟨Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row
        columnMap sourceRow,
      List.mem_map.2 ⟨sourceRow, rowMember, rfl⟩,
      ?_⟩
  rw [
    Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row_columnIds]
  unfold numericColumnIds at mentioned
  rcases List.mem_map.1 mentioned with
    ⟨term, termMember, sourceExact⟩
  exact List.mem_map.2
    ⟨term, termMember, by simpa only [sourceExact]⟩

/-- Canonical R1CS coverage is the same column-use fact expressed through
`Mentions`; expose it to the selected lowering layer without changing rows. -/
theorem of_rows_cover
    (rows : List NumericRow) (columns : List Nat)
    (covered :
      Nightstream.Implementation.R1CS.Canonical.AllocationCoverage.RowsCover
        rows columns) :
    NumericRowsCover rows columns := by
  intro column member
  rcases covered column member with
    ⟨row, rowMember, mentioned⟩
  refine ⟨row, rowMember, ?_⟩
  unfold numericColumnIds
  rcases mentioned with inA | inB | inC
  · unfold LinCombNormal.Mentions at inA
    simp only [List.map_append, List.mem_append]
    exact Or.inl (Or.inl inA)
  · unfold LinCombNormal.Mentions at inB
    simp only [List.map_append, List.mem_append]
    exact Or.inl (Or.inr inB)
  · unfold LinCombNormal.Mentions at inC
    simp only [List.map_append, List.mem_append]
    exact Or.inr inC

/-- A structured outer `flatMap` preserves per-part coverage. -/
theorem numeric_flatMap
    {α : Type}
    (parts : List α)
    (rows : α → List NumericRow)
    (columns : α → List Nat)
    (covered : ∀ part ∈ parts, NumericRowsCover (rows part) (columns part)) :
    NumericRowsCover (parts.flatMap rows) (parts.flatMap columns) := by
  intro column member
  rcases List.mem_flatMap.1 member with
    ⟨part, partMember, columnMember⟩
  rcases covered part partMember column columnMember with
    ⟨row, rowMember, mentioned⟩
  exact ⟨row,
    List.mem_flatMap.2 ⟨part, partMember, rowMember⟩,
    mentioned⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage
