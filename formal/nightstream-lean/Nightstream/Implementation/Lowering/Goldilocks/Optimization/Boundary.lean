import Nightstream.Implementation.Lowering.Goldilocks.Compiler

/-!
Contract: the stable physical interface observed by constraint replacements.

Assurance tier: model-level.

Owns: ordered committed, public, output, and transcript-protected column
projections.

Does not own: which columns a protocol selects, semantic event transport,
constraint satisfaction, or Rust.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Boundary

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

private abbrev Field := Nightstream.SuperNeo.Concrete.F
abbrev Assignment := ColumnId -> Field

/-- Physical columns that a replacement must preserve in exact order.

`outputs` and `transcript` may repeat committed or public columns. Keeping the
roles separate makes the review boundary explicit and prevents a public-role
count from standing in for transcript preservation. -/
structure Columns where
  committedColumns : List ColumnId
  publicColumns : List ColumnId
  outputColumns : List ColumnId
  transcriptColumns : List ColumnId
deriving DecidableEq, Repr

/-- Concrete observed values at the selected boundary. -/
structure Values where
  committedValues : List Field
  publicValues : List Field
  outputValues : List Field
  transcriptValues : List Field
deriving DecidableEq, Repr

def values (columns : Columns) (assignment : Assignment) : Values where
  committedValues := columns.committedColumns.map assignment
  publicValues := columns.publicColumns.map assignment
  outputValues := columns.outputColumns.map assignment
  transcriptValues := columns.transcriptColumns.map assignment

private def idsWithOwnership
    (ownership : Ownership)
    (columns : List OwnedColumn) : List ColumnId :=
  (columns.filter fun column => decide (column.ownership = ownership)).map
    (fun column => column.id)

/-- Build a boundary from any exact physical allocation stream. -/
def ofOwnedColumns
    (columns : List OwnedColumn)
    (outputs transcript : List ColumnId) : Columns where
  committedColumns := idsWithOwnership .committedColumn columns
  publicColumns := idsWithOwnership .publicColumn columns
  outputColumns := outputs
  transcriptColumns := transcript

/-- Default external boundary from the exact physical allocation stream.

Output and transcript protection are supplied separately because ownership
classes alone do not identify those semantic roles. -/
def ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (outputs transcript : List ColumnId) : Columns :=
  ofOwnedColumns encoding.columns outputs transcript

theorem values_eq_of_agrees
    (columns : Columns)
    (left right : Assignment)
    (agrees :
      forall column,
        column ∈
            columns.committedColumns ++ columns.publicColumns ++
              columns.outputColumns ++ columns.transcriptColumns ->
          left column = right column) :
    values columns left = values columns right := by
  cases columns with
  | mk committedColumns publicColumns outputColumns transcriptColumns =>
      have committedValuesEq :
          committedColumns.map left = committedColumns.map right := by
        apply List.map_congr_left
        intro column member
        apply agrees column
        simpa using
          (List.mem_append_left transcriptColumns
            (List.mem_append_left outputColumns
              (List.mem_append_left publicColumns member)))
      have publicValuesEq :
          publicColumns.map left = publicColumns.map right := by
        apply List.map_congr_left
        intro column member
        apply agrees column
        simpa using
          (List.mem_append_left transcriptColumns
            (List.mem_append_left outputColumns
              (List.mem_append_right committedColumns member)))
      have outputValuesEq :
          outputColumns.map left = outputColumns.map right := by
        apply List.map_congr_left
        intro column member
        apply agrees column
        simpa using
          (List.mem_append_left transcriptColumns
            (List.mem_append_right (committedColumns ++ publicColumns) member))
      have transcriptValuesEq :
          transcriptColumns.map left = transcriptColumns.map right := by
        apply List.map_congr_left
        intro column member
        apply agrees column
        simpa using
          (List.mem_append_right
            ((committedColumns ++ publicColumns) ++ outputColumns) member)
      simp only [values]
      rw [committedValuesEq, publicValuesEq, outputValuesEq,
        transcriptValuesEq]

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Boundary
