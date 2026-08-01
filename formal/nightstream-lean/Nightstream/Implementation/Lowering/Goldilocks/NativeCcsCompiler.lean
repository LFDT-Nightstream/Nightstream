import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.StableRows

/-!
Contract: compile one receipt-conserving native selected-CCS program into the
exact four finite CCS matrices.

Assurance tier: model-level.

Owns:
- the canonical finite index of every allocated structural column;
- exact A, B, C, and selector matrix rows;
- zero padding to the selected Boolean row domain;
- soundness and honest completeness against the source native program;
- exact logical matrix width and row occurrence preservation.

Does not own: Phi81 carrier completion, a concrete F-prime deployment, JSON,
Rust parsing, commitments, or a security reduction.

Emits constraints: no new rows or columns. It converts each selected source
row into one row across four matrices.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- Static facts required before a structural program can become a finite
matrix program. No fallback index is load-bearing under this predicate. -/
structure Valid (program : NativeCcsProgram.Program) : Prop where
  oneAllocated : program.one ∈ program.columnIds
  columnIdsNodup : program.columnIds.Nodup
  rowsSupported :
    ∀ row, row ∈ program.rows →
      ∀ column, column ∈ row.columnIds →
        column ∈ program.columnIds

/-- One Boolean row domain that contains every emitted row. -/
structure RowDomain (program : NativeCcsProgram.Program) where
  rowVariables : Nat
  rowsCovered : program.rows.length ≤ 2 ^ rowVariables

namespace ColumnIndex

/-- A proof-carrying position in the exact ordered allocation list. -/
structure Location (columns : List ColumnId) (column : ColumnId) where
  index : Fin columns.length
  atIndex : columns.get index = column

/-- Locate the first structural occurrence. `Valid.columnIdsNodup` later
proves that it is also the only occurrence. -/
def locate
    (columns : List ColumnId)
    (column : ColumnId)
    (member : column ∈ columns) :
    Location columns column := by
  match found : columns.idxOf? column with
  | none =>
      exact False.elim ((List.idxOf?_eq_none_iff.mp found) member)
  | some index =>
      have witness := List.idxOf?_eq_some_iff.mp found
      exact ⟨⟨index, Exists.elim witness fun bound _ => bound⟩,
        Exists.elim witness fun _ exactAndFirst => exactAndFirst.1⟩

/-- Total structural-to-finite map. The fallback is the allocated constant
one position. `Valid.rowsSupported` proves that emitted data never uses it. -/
def index
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (column : ColumnId) :
    Fin program.columnIds.length :=
  if member : column ∈ program.columnIds then
    (locate program.columnIds column member).index
  else
    (locate program.columnIds program.one valid.oneAllocated).index

def columnAt
    (program : NativeCcsProgram.Program)
    (column : Fin program.columnIds.length) : ColumnId :=
  program.columnIds.get column

theorem columnAt_index
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (column : ColumnId)
    (member : column ∈ program.columnIds) :
    columnAt program (index program valid column) = column := by
  unfold index
  rw [dif_pos member]
  exact (locate program.columnIds column member).atIndex

end ColumnIndex

/-- Read a finite assignment through the canonical structural index. -/
def pulledAssignment
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (assignment : Fin program.columnIds.length → Field) :
    ColumnId → Field :=
  StableRows.pulledAssignment (ColumnIndex.index program valid) assignment

/-- Rebuild the finite assignment in exact allocation order. -/
def indexedAssignment
    (program : NativeCcsProgram.Program)
    (assignment : ColumnId → Field) :
    Fin program.columnIds.length → Field :=
  fun column => assignment (ColumnIndex.columnAt program column)

theorem pulled_indexed_at_allocated
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (assignment : ColumnId → Field)
    (column : ColumnId)
    (member : column ∈ program.columnIds) :
    pulledAssignment program valid (indexedAssignment program assignment)
        column =
      assignment column := by
  unfold pulledAssignment StableRows.pulledAssignment indexedAssignment
  rw [ColumnIndex.columnAt_index program valid column member]

/-- Exact matrix-image tuple for one selected source row. -/
def rowPoint
    (row : SelectedRow)
    (assignment : ColumnId → Field) :
    Fin NativeCcsSelector.matrixCount → Field :=
  fun matrix =>
    if matrix.val = 0 then row.source.row.a.eval assignment
    else if matrix.val = 1 then row.source.row.b.eval assignment
    else if matrix.val = 2 then row.source.row.c.eval assignment
    else assignment row.selector

/-- One dense matrix row. Matrix order is exactly `[A, B, C, S]`. -/
def matrixRow
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (row : SelectedRow)
    (matrix : Fin NativeCcsSelector.matrixCount) :
    DirectRows.LinearCombination program.columnIds.length :=
  if matrix.val = 0 then
    StableRows.denseCombination (ColumnIndex.index program valid)
      row.source.row.a
  else if matrix.val = 1 then
    StableRows.denseCombination (ColumnIndex.index program valid)
      row.source.row.b
  else if matrix.val = 2 then
    StableRows.denseCombination (ColumnIndex.index program valid)
      row.source.row.c
  else
    StableRows.denseCombination (ColumnIndex.index program valid)
      (singleton row.selector 1)

theorem matrixRow_eval
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (row : SelectedRow)
    (matrix : Fin NativeCcsSelector.matrixCount)
    (assignment : Fin program.columnIds.length → Field) :
    DirectRows.LinearCombination.eval
        (matrixRow program valid row matrix) assignment =
      rowPoint row (pulledAssignment program valid assignment) matrix := by
  have matrixLt : matrix.val < 4 := by
    simpa [NativeCcsSelector.matrixCount] using matrix.isLt
  unfold matrixRow rowPoint
  by_cases zero : matrix.val = 0
  · simp only [zero, if_pos]
    exact StableRows.denseCombination_eval
      (ColumnIndex.index program valid) row.source.row.a assignment
  · rw [if_neg zero]
    by_cases one : matrix.val = 1
    · simp only [one, if_pos]
      exact StableRows.denseCombination_eval
        (ColumnIndex.index program valid) row.source.row.b assignment
    · rw [if_neg one]
      by_cases two : matrix.val = 2
      · simp only [two, if_pos]
        exact StableRows.denseCombination_eval
          (ColumnIndex.index program valid) row.source.row.c assignment
      · rw [if_neg two]
        rw [StableRows.denseCombination_eval]
        simp [singleton, LinearCombination.eval, pulledAssignment,
          zero, one, two, Fin.one_mul]

/-- Four finite numeric matrices before Boolean-row padding. -/
def finiteMatrices
    (program : NativeCcsProgram.Program)
    (valid : Valid program) :
    Fin NativeCcsSelector.matrixCount →
      RowPadding.NumericMatrix Field program.rows.length
        program.columnIds.length :=
  fun matrix row =>
    matrixRow program valid (program.rows.get row) matrix

/-- Paper shape of the exact four-matrix finite relation. Batch counts are
zero because this object is only the source CCS relation. -/
def paperShape
    (program : NativeCcsProgram.Program)
    (domain : RowDomain program) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape :=
  Phi81MatrixSource.phi81Shape domain.rowVariables 0 0
    NativeCcsSelector.matrixCount

/-- Exact Boolean-row-padded CCS system at the program's logical width. -/
def system
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program) :
    CCSResidualTable.Structure Field (paperShape program domain)
      program.columnIds.length where
  matrices := fun matrix =>
    RowPadding.padRows (finiteMatrices program valid matrix)
  constraintPolynomial := NativeCcsSelector.constraintPolynomial

theorem matrixImagesAt_sourceRow
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field)
    (row : Fin program.rows.length) :
    matrixImagesAt baseOps (system program valid domain) assignment
        (RowPadding.numericRowVertex domain.rowsCovered row) =
      rowPoint (program.rows.get row)
        (pulledAssignment program valid assignment) := by
  funext matrix
  unfold matrixImagesAt system finiteMatrices
  rw [DirectRows.matrixVectorAt_padRows_numeric]
  exact matrixRow_eval program valid (program.rows.get row) matrix assignment

theorem residualAt_sourceRow
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field)
    (row : Fin program.rows.length) :
    residualAt baseOps (system program valid domain) assignment
        (RowPadding.numericRowVertex domain.rowsCovered row) =
      NativeCcsSelector.polynomial
        ((program.rows.get row).source.row.a.eval
          (pulledAssignment program valid assignment))
        ((program.rows.get row).source.row.b.eval
          (pulledAssignment program valid assignment))
        ((program.rows.get row).source.row.c.eval
          (pulledAssignment program valid assignment))
        ((pulledAssignment program valid assignment)
          (program.rows.get row).selector) := by
  unfold residualAt
  change
    NativeCcsSelector.evaluate
        (matrixImagesAt baseOps (system program valid domain) assignment
          (RowPadding.numericRowVertex domain.rowsCovered row)) =
      _
  rw [matrixImagesAt_sourceRow, NativeCcsSelector.evaluate_exact]
  simp [rowPoint, NativeCcsSelector.matrixCount]

theorem residualAt_sourceRow_eq_zero_iff
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field)
    (row : Fin program.rows.length) :
    residualAt baseOps (system program valid domain) assignment
        (RowPadding.numericRowVertex domain.rowsCovered row) = 0 ↔
      (program.rows.get row).Holds
        (pulledAssignment program valid assignment) := by
  rw [residualAt_sourceRow]
  rfl

theorem matrixImagesAt_padding
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field)
    (vertex : BooleanVertex domain.rowVariables)
    (padding : program.rows.length ≤ rowIndex vertex) :
    matrixImagesAt baseOps (system program valid domain) assignment vertex =
      fun _ => 0 := by
  funext matrix
  unfold matrixImagesAt system
  exact DirectRows.matrixVectorAt_padRows_padding
    (finiteMatrices program valid matrix) assignment vertex padding

theorem residualAt_padding
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field)
    (vertex : BooleanVertex domain.rowVariables)
    (padding : program.rows.length ≤ rowIndex vertex) :
    residualAt baseOps (system program valid domain) assignment vertex = 0 := by
  unfold residualAt
  rw [matrixImagesAt_padding program valid domain assignment vertex padding]
  change NativeCcsSelector.evaluate (fun _ => 0) = 0
  rw [NativeCcsSelector.evaluate_exact]
  rfl

private theorem satisfies_iff_forall_index
    (rows : List SelectedRow)
    (assignment : ColumnId → Field) :
    NativeCcsSelector.Satisfies rows assignment ↔
      ∀ row : Fin rows.length, (rows.get row).Holds assignment := by
  induction rows with
  | nil =>
      constructor
      · intro _ row
        exact Fin.elim0 row
      · intro _
        trivial
  | cons head tail inductionHypothesis =>
      constructor
      · intro satisfied row
        refine Fin.cases ?_ (fun tailRow => ?_) row
        · exact satisfied.1
        · exact inductionHypothesis.mp satisfied.2 tailRow
      · intro all
        exact ⟨
          all ⟨0, by simp⟩,
          inductionHypothesis.mpr
            (fun row => by simpa using all (Fin.succ row))
        ⟩

/-- Finite matrix soundness and honest completeness. The same indexed
assignment is used on both sides. -/
theorem constraintSatisfied_iff
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field) :
    ConstraintSatisfied baseOps (system program valid domain) assignment ↔
      NativeCcsSelector.Satisfies program.rows
        (pulledAssignment program valid assignment) := by
  rw [satisfies_iff_forall_index]
  constructor
  · intro satisfied row
    exact
      (residualAt_sourceRow_eq_zero_iff
        program valid domain assignment row).mp
        (satisfied
          (RowPadding.numericRowVertex domain.rowsCovered row))
  · intro sourceHolds vertex
    by_cases live : rowIndex vertex < program.rows.length
    · let row : Fin program.rows.length := ⟨rowIndex vertex, live⟩
      have vertexEq :
          RowPadding.numericRowVertex domain.rowsCovered row = vertex := by
        unfold RowPadding.numericRowVertex
        simpa [row] using rowVertex_rowIndex vertex
      rw [← vertexEq]
      exact
        (residualAt_sourceRow_eq_zero_iff
          program valid domain assignment row).mpr
          (sourceHolds row)
    · exact residualAt_padding program valid domain assignment vertex
        (Nat.le_of_not_gt live)

/-- The finite verifier boundary includes the canonical constant-one
coordinate. -/
def IndexedAccepts
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field) : Prop :=
  assignment (ColumnIndex.index program valid program.one) = 1 ∧
    ConstraintSatisfied baseOps (system program valid domain) assignment

theorem indexedAccepts_iff
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : Fin program.columnIds.length → Field) :
    IndexedAccepts program valid domain assignment ↔
      program.Satisfies (pulledAssignment program valid assignment) := by
  unfold IndexedAccepts NativeCcsProgram.Program.Satisfies pulledAssignment
    StableRows.pulledAssignment
  rw [constraintSatisfied_iff]
  rfl

private theorem combination_eval_congr
    (combination : LinearCombination)
    (left right : ColumnId → Field)
    (agree :
      ∀ term, term ∈ combination →
        left term.column = right term.column) :
    combination.eval left = combination.eval right := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      simp only [LinearCombination.eval]
      rw [agree term List.mem_cons_self]
      rw [inductionHypothesis (fun candidate member =>
        agree candidate (List.mem_cons_of_mem term member))]

private theorem selectedRow_holds_congr
    (row : SelectedRow)
    (left right : ColumnId → Field)
    (agree :
      ∀ column, column ∈ row.columnIds → left column = right column) :
    row.Holds left ↔ row.Holds right := by
  have selectorEqual :
      left row.selector = right row.selector :=
    agree row.selector List.mem_cons_self
  have sourceAgree :
      ∀ column, column ∈ row.source.columnIds →
        left column = right column := by
    intro column member
    exact agree column (List.mem_cons_of_mem row.selector member)
  have aEqual := combination_eval_congr row.source.row.a left right
    (fun term member =>
      sourceAgree term.column
        (by
          unfold OwnedRow.columnIds Row.columnIds
          exact List.mem_map.mpr
            ⟨term,
              List.mem_append_left _
                (List.mem_append_left _ member),
              rfl⟩))
  have bEqual := combination_eval_congr row.source.row.b left right
    (fun term member =>
      sourceAgree term.column
        (by
          unfold OwnedRow.columnIds Row.columnIds
          exact List.mem_map.mpr
            ⟨term,
              List.mem_append_left _
                (List.mem_append_right _ member),
              rfl⟩))
  have cEqual := combination_eval_congr row.source.row.c left right
    (fun term member =>
      sourceAgree term.column
        (by
          unfold OwnedRow.columnIds Row.columnIds
          exact List.mem_map.mpr
            ⟨term,
              List.mem_append_right _ member,
              rfl⟩))
  unfold SelectedRow.Holds
  rw [aEqual, bEqual, cEqual, selectorEqual]

private theorem satisfies_congr
    (rows : List SelectedRow)
    (left right : ColumnId → Field)
    (agree :
      ∀ row, row ∈ rows →
        ∀ column, column ∈ row.columnIds →
          left column = right column) :
    NativeCcsSelector.Satisfies rows left ↔
      NativeCcsSelector.Satisfies rows right := by
  induction rows with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      rw [NativeCcsSelector.satisfies_cons,
        NativeCcsSelector.satisfies_cons]
      rw [selectedRow_holds_congr head left right
        (fun column member =>
          agree head List.mem_cons_self column member)]
      rw [inductionHypothesis (fun row member column columnMember =>
        agree row (List.mem_cons_of_mem head member) column columnMember)]

theorem pulled_indexed_satisfies_iff
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (assignment : ColumnId → Field) :
    program.Satisfies
        (pulledAssignment program valid
          (indexedAssignment program assignment)) ↔
      program.Satisfies assignment := by
  unfold NativeCcsProgram.Program.Satisfies
  have oneEqual :=
    pulled_indexed_at_allocated program valid assignment program.one
      valid.oneAllocated
  rw [oneEqual]
  exact and_congr_right fun _ =>
    satisfies_congr program.rows
      (pulledAssignment program valid
        (indexedAssignment program assignment))
      assignment
      (fun row rowMember column columnMember =>
        pulled_indexed_at_allocated program valid assignment column
          (valid.rowsSupported row rowMember column columnMember))

/-- Honest structural assignments reassemble into the exact indexed
assignment, and no finite-matrix acceptance is added or lost. -/
theorem indexedAssignment_accepts_iff
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program)
    (assignment : ColumnId → Field) :
    IndexedAccepts program valid domain
        (indexedAssignment program assignment) ↔
      program.Satisfies assignment := by
  rw [indexedAccepts_iff, pulled_indexed_satisfies_iff]

theorem matrix_count_exact
    (program : NativeCcsProgram.Program)
    (valid : Valid program)
    (domain : RowDomain program) :
    (system program valid domain).constraintPolynomial =
      NativeCcsSelector.constraintPolynomial :=
  rfl

theorem logical_width_exact
    (program : NativeCcsProgram.Program) :
    program.columnIds.length =
      program.cost.committedColumns +
        program.cost.publicColumns +
        program.cost.auxiliaryColumns :=
  NativeCcsProgram.Program.columnIds_length_eq_cost_columns program

end Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler
