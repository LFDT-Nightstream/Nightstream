import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedLayout
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedPolynomial
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler
import Nightstream.Implementation.Lowering.Nebula.Physical

/-!
Finite CCS compiler for the Nebula-enabled 42-times-6 relation.

Assurance tier: model-level.

This file owns the exact row and column composition. Native F-prime rows
occupy the first row interval. The selected Nebula rows occupy the next
interval. Each family is zero in the other family's matrix roles. The
combined polynomial therefore reduces to the source polynomial on every
live row.

It does not own the recursive fixed point, application codecs, transcript
binding, an Ajtai key, a Rust manifest, or a security reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 20000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedCompiler

open Nightstream.Implementation.Lowering
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedLayout
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedPolynomial
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler

private abbrev Field := Nightstream.SuperNeo.Concrete.F
private abbrev Dense (columns : Nat) :=
  DirectRows.LinearCombination columns

/-- Source data required to compile one native program together with the
selected standalone Nebula program. -/
structure Source (program : NativeCcsProgram.Program) where
  nativeValid : NativeCcsCompiler.Valid program
  nativePublicFits : NebulaFreshCarrier.linkWidth <= program.columnIds.length
  rowVariables : Nat
  rowsCovered :
    program.rows.length +
        (Nebula.Compiler.rows Nebula.Layout.wasm42x6).length <=
      2 ^ rowVariables

namespace Source

def dimensions {program : NativeCcsProgram.Program}
    (source : Source program) : NebulaCombinedLayout.Dimensions where
  rowVariables := source.rowVariables
  nativeLogicalWidth := program.columnIds.length
  nativePublicFits := source.nativePublicFits

def rowCount {program : NativeCcsProgram.Program}
    (_source : Source program) : Nat :=
  program.rows.length + (Nebula.Compiler.rows Nebula.Layout.wasm42x6).length

def columnCount {program : NativeCcsProgram.Program}
    (source : Source program) : Nat :=
  source.dimensions.logicalWidth

theorem rowCount_eq {program : NativeCcsProgram.Program}
    (source : Source program) :
    source.rowCount = program.rows.length + 422465 := by
  unfold rowCount
  rw [Nebula.Compiler.wasm42x6_rows_length]

end Source

/-- Native structural columns placed in the combined assignment. -/
def nativeIndex
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (column : ColumnId) : Fin source.columnCount :=
  NebulaCombinedLayout.nativeColumn source.dimensions
    (NativeCcsCompiler.ColumnIndex.index program source.nativeValid column)

/-- A total numeric placement. The fallback is the shared constant. Physical
support proves that no selected Nebula row uses the fallback. -/
def nebulaIndex
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (column : Nat) : Fin source.columnCount :=
  if bounded : column < Nebula.Layout.wasm42x6.columnCount then
    NebulaCombinedLayout.nebulaColumn source.dimensions ⟨column, bounded⟩
  else
    ⟨0, Nat.lt_of_lt_of_le (by decide) source.dimensions.publicFits⟩

/-- Native assignment read through the exact structural placement. -/
def nativeAssignment
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (assignment : Fin source.columnCount -> Field) : ColumnId -> Field :=
  fun column => assignment (nativeIndex source column)

/-- Nebula assignment read through the exact numeric placement. -/
def nebulaAssignment
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (assignment : Fin source.columnCount -> Field) : Nat -> Field :=
  fun column => assignment (nebulaIndex source column)

private def numericColumnId (column : Nat) : ColumnId where
  owner := .prelude
  bundleIndex := 1
  coordinateIndex := column

private def numericCombination
    (combination : Nebula.Rows.LinearCombination) :
    Goldilocks.LinearCombination :=
  combination.map fun term =>
    { column := numericColumnId term.column
      coefficient := term.coefficient }

private def numericIndex
    {program : NativeCcsProgram.Program}
    (source : Source program) (column : ColumnId) : Fin source.columnCount :=
  nebulaIndex source column.coordinateIndex

private theorem numericCombination_eval
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (combination : Nebula.Rows.LinearCombination)
    (assignment : Fin source.columnCount -> Field) :
    Goldilocks.LinearCombination.eval
        (StableRows.pulledAssignment (numericIndex source) assignment)
        (numericCombination combination) =
      Nebula.Rows.LinearCombination.eval
        (nebulaAssignment source assignment) combination := by
  induction combination with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      change
        term.coefficient *
              StableRows.pulledAssignment (numericIndex source) assignment
                (numericColumnId term.column) +
            Goldilocks.LinearCombination.eval
              (StableRows.pulledAssignment (numericIndex source) assignment)
              (numericCombination rest) =
          term.coefficient * nebulaAssignment source assignment term.column +
            Nebula.Rows.LinearCombination.eval
              (nebulaAssignment source assignment) rest
      rw [inductionHypothesis]
      rfl

private def nativeSparseFor
    (row : NativeCcsSelector.SelectedRow)
    (matrix : Fin NativeCcsSelector.matrixCount) :
    Goldilocks.LinearCombination :=
  if matrix.val = 0 then row.source.row.a
  else if matrix.val = 1 then row.source.row.b
  else if matrix.val = 2 then row.source.row.c
  else singleton row.selector 1

def nativeDenseRow
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : NativeCcsSelector.SelectedRow)
    (matrix : Fin NebulaCombinedPolynomial.matrixCount) :
    Dense source.columnCount :=
  if nativeRole : matrix.val < NativeCcsSelector.matrixCount then
    StableRows.denseCombination (nativeIndex source)
      (nativeSparseFor row ⟨matrix.val, nativeRole⟩)
  else
    fun _ => 0

/-- Dense combined row for one Nebula source row and one combined matrix
role. Native roles are zero. -/
def nebulaDenseRow
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Nebula.Rows.Row)
    (matrix : Fin NebulaCombinedPolynomial.matrixCount) :
    Dense source.columnCount :=
  if nebulaRole : NativeCcsSelector.matrixCount <= matrix.val then
    let index : Fin Nebula.StepPolynomial.matrixCount :=
      ⟨matrix.val - NativeCcsSelector.matrixCount, by
        have matrixBound := matrix.isLt
        simp only [NebulaCombinedPolynomial.matrixCount,
          NativeCcsSelector.matrixCount, Nebula.StepPolynomial.matrixCount]
          at matrixBound ⊢
        omega⟩
    StableRows.denseCombination (numericIndex source)
      (numericCombination (row.images.at (Nebula.StepPolynomial.Role.ofIndex index)))
  else
    fun _ => 0

/-- Exact finite combined matrices before Boolean-row padding. -/
def finiteMatrices
    {program : NativeCcsProgram.Program}
    (source : Source program) :
    Fin NebulaCombinedPolynomial.matrixCount ->
      RowPadding.NumericMatrix Field source.rowCount source.columnCount :=
  fun matrix row =>
    if nativeRow : row.val < program.rows.length then
      nativeDenseRow source (program.rows.get ⟨row.val, nativeRow⟩) matrix
    else
      nebulaDenseRow source
        ((Nebula.Compiler.rows Nebula.Layout.wasm42x6).get
          ⟨row.val - program.rows.length, by
            have rowBound := row.isLt
            simp only [Source.rowCount] at rowBound
            omega⟩)
        matrix

/-- Paper shape of the exact combined matrix family. -/
def paperShape
    {program : NativeCcsProgram.Program}
    (source : Source program) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape :=
  Phi81MatrixSource.phi81Shape source.rowVariables 0 0
    NebulaCombinedPolynomial.matrixCount

/-- Exact Boolean-row-padded nineteen-matrix relation. -/
def system
    {program : NativeCcsProgram.Program}
    (source : Source program) :
    CCSResidualTable.Structure Field (paperShape source) source.columnCount where
  matrices := fun matrix => RowPadding.padRows (finiteMatrices source matrix)
  constraintPolynomial := NebulaCombinedPolynomial.polynomial

/-- Position of one native row in the combined row interval. -/
def nativeRowPosition
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Fin program.rows.length) : Fin source.rowCount :=
  ⟨row.val, by
    have bound := row.isLt
    unfold Source.rowCount
    omega⟩

/-- Position of one Nebula row after the native row interval. -/
def nebulaRowPosition
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Fin (Nebula.Compiler.rows Nebula.Layout.wasm42x6).length) :
    Fin source.rowCount :=
  ⟨program.rows.length + row.val, by
    have bound := row.isLt
    unfold Source.rowCount
    omega⟩

/-- Matrix-image point emitted for one native row. -/
def nativeCombinedPoint
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : NativeCcsSelector.SelectedRow)
    (assignment : Fin source.columnCount -> Field) :
    Fin NebulaCombinedPolynomial.matrixCount -> Field :=
  fun matrix =>
    if nativeRole : matrix.val < NativeCcsSelector.matrixCount then
      Goldilocks.LinearCombination.eval (nativeAssignment source assignment)
        (nativeSparseFor row ⟨matrix.val, nativeRole⟩)
    else
      0

/-- Matrix-image point emitted for one Nebula row. -/
def nebulaCombinedPoint
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Nebula.Rows.Row)
    (assignment : Fin source.columnCount -> Field) :
    Fin NebulaCombinedPolynomial.matrixCount -> Field :=
  fun matrix =>
    if nebulaRole : NativeCcsSelector.matrixCount <= matrix.val then
      row.point (nebulaAssignment source assignment)
        ⟨matrix.val - NativeCcsSelector.matrixCount, by
          have matrixBound := matrix.isLt
          simp only [NebulaCombinedPolynomial.matrixCount] at matrixBound ⊢
          omega⟩
    else
      0

theorem nativeDenseRow_eval
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : NativeCcsSelector.SelectedRow)
    (matrix : Fin NebulaCombinedPolynomial.matrixCount)
    (nativeRole : matrix.val < NativeCcsSelector.matrixCount)
    (assignment : Fin source.columnCount -> Field) :
    DirectRows.LinearCombination.eval
        (nativeDenseRow source row matrix) assignment =
      Goldilocks.LinearCombination.eval
        (nativeAssignment source assignment)
        (nativeSparseFor row ⟨matrix.val, nativeRole⟩) := by
  simp only [nativeDenseRow, dif_pos nativeRole]
  exact StableRows.denseCombination_eval
    (nativeIndex source) (nativeSparseFor row ⟨matrix.val, nativeRole⟩)
      assignment

theorem nativeSparseFor_eval
    (row : NativeCcsSelector.SelectedRow)
    (assignment : ColumnId -> Field)
    (index : Fin NativeCcsSelector.matrixCount) :
    Goldilocks.LinearCombination.eval assignment
        (nativeSparseFor row index) =
      NativeCcsCompiler.rowPoint row assignment index := by
  rcases index with ⟨index, bound⟩
  change index < 4 at bound
  by_cases zero : index = 0
  · subst index
    rfl
  by_cases one : index = 1
  · subst index
    simp [nativeSparseFor, NativeCcsCompiler.rowPoint]
  by_cases two : index = 2
  · subst index
    simp [nativeSparseFor, NativeCcsCompiler.rowPoint]
  have three : index = 3 := by omega
  subst index
  simp [nativeSparseFor, NativeCcsCompiler.rowPoint, Goldilocks.singleton,
    Goldilocks.LinearCombination.eval, Fin.one_mul]

theorem nativePoint_nativeCombinedPoint
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : NativeCcsSelector.SelectedRow)
    (assignment : Fin source.columnCount -> Field) :
    NebulaCombinedPolynomial.nativePoint
        (nativeCombinedPoint source row assignment) =
      NativeCcsCompiler.rowPoint row
        (nativeAssignment source assignment) := by
  funext index
  simp only [NebulaCombinedPolynomial.nativePoint,
    NebulaCombinedPolynomial.nativeIndex, nativeCombinedPoint]
  rw [dif_pos index.isLt]
  exact nativeSparseFor_eval row (nativeAssignment source assignment) index

theorem nebulaPoint_nativeCombinedPoint_zero
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : NativeCcsSelector.SelectedRow)
    (assignment : Fin source.columnCount -> Field)
    (index : Fin Nebula.StepPolynomial.matrixCount) :
    NebulaCombinedPolynomial.nebulaPoint
        (nativeCombinedPoint source row assignment) index = 0 := by
  unfold NebulaCombinedPolynomial.nebulaPoint
  simp only [NebulaCombinedPolynomial.nebulaIndex, nativeCombinedPoint]
  rw [dif_neg]
  have bound := index.isLt
  omega

theorem nebulaDenseRow_eval
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Nebula.Rows.Row)
    (matrix : Fin NebulaCombinedPolynomial.matrixCount)
    (nebulaRole : NativeCcsSelector.matrixCount <= matrix.val)
    (assignment : Fin source.columnCount -> Field) :
    DirectRows.LinearCombination.eval
        (nebulaDenseRow source row matrix) assignment =
      Nebula.Rows.LinearCombination.eval
        (nebulaAssignment source assignment)
        (row.images.at
          (Nebula.StepPolynomial.Role.ofIndex
            ⟨matrix.val - NativeCcsSelector.matrixCount, by
              have matrixBound := matrix.isLt
              simp only [NebulaCombinedPolynomial.matrixCount,
                NativeCcsSelector.matrixCount,
                Nebula.StepPolynomial.matrixCount] at matrixBound ⊢
              omega⟩)) := by
  simp only [nebulaDenseRow, dif_pos nebulaRole]
  rw [StableRows.denseCombination_eval]
  exact numericCombination_eval source _ assignment

theorem nativePoint_nebulaCombinedPoint_zero
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Nebula.Rows.Row)
    (assignment : Fin source.columnCount -> Field)
    (index : Fin NativeCcsSelector.matrixCount) :
    NebulaCombinedPolynomial.nativePoint
        (nebulaCombinedPoint source row assignment) index = 0 := by
  unfold NebulaCombinedPolynomial.nativePoint
  simp only [NebulaCombinedPolynomial.nativeIndex, nebulaCombinedPoint]
  rw [dif_neg]
  exact Nat.not_le_of_gt index.isLt

theorem nebulaPoint_nebulaCombinedPoint
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (row : Nebula.Rows.Row)
    (assignment : Fin source.columnCount -> Field) :
    NebulaCombinedPolynomial.nebulaPoint
        (nebulaCombinedPoint source row assignment) =
      row.point (nebulaAssignment source assignment) := by
  funext index
  unfold NebulaCombinedPolynomial.nebulaPoint
  simp only [NebulaCombinedPolynomial.nebulaIndex, nebulaCombinedPoint]
  rw [dif_pos (by omega)]
  apply congrArg (row.point (nebulaAssignment source assignment))
  apply Fin.ext
  simp

theorem matrixImagesAt_nativeRow
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (assignment : Fin source.columnCount -> Field)
    (row : Fin program.rows.length) :
    matrixImagesAt baseOps (system source) assignment
        (RowPadding.numericRowVertex source.rowsCovered
          (nativeRowPosition source row)) =
      nativeCombinedPoint source (program.rows.get row) assignment := by
  funext matrix
  unfold matrixImagesAt system
  simp only
  unfold matrixVectorAt
  change
    DirectRows.LinearCombination.eval
        (finiteMatrices source matrix (nativeRowPosition source row))
        assignment =
      nativeCombinedPoint source (program.rows.get row) assignment matrix
  have finiteRow :
      finiteMatrices source matrix (nativeRowPosition source row) =
        nativeDenseRow source (program.rows.get row) matrix := by
    simp [finiteMatrices, nativeRowPosition, row.isLt]
  rw [finiteRow]
  by_cases nativeRole : matrix.val < NativeCcsSelector.matrixCount
  · rw [nativeDenseRow_eval source (program.rows.get row) matrix
      nativeRole assignment]
    rfl
  · simp [nativeDenseRow, nativeCombinedPoint, nativeRole]

theorem residualAt_nativeRow
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (assignment : Fin source.columnCount -> Field)
    (row : Fin program.rows.length) :
    residualAt baseOps (system source) assignment
        (RowPadding.numericRowVertex source.rowsCovered
          (nativeRowPosition source row)) =
      NativeCcsSelector.evaluate
        (NativeCcsCompiler.rowPoint (program.rows.get row)
          (nativeAssignment source assignment)) := by
  unfold residualAt
  change
    NebulaCombinedPolynomial.evaluate
        (matrixImagesAt baseOps (system source) assignment
          (RowPadding.numericRowVertex source.rowsCovered
            (nativeRowPosition source row))) = _
  rw [matrixImagesAt_nativeRow]
  rw [NebulaCombinedPolynomial.evaluate_native_only]
  · rw [nativePoint_nativeCombinedPoint]
  · exact nebulaPoint_nativeCombinedPoint_zero source
      (program.rows.get row) assignment

theorem residualAt_nativeRow_eq_zero_iff
    {program : NativeCcsProgram.Program}
    (source : Source program)
    (assignment : Fin source.columnCount -> Field)
    (row : Fin program.rows.length) :
    residualAt baseOps (system source) assignment
        (RowPadding.numericRowVertex source.rowsCovered
          (nativeRowPosition source row)) = 0 <->
      (program.rows.get row).Holds (nativeAssignment source assignment) := by
  rw [residualAt_nativeRow]
  rw [NativeCcsSelector.evaluate_exact]
  rfl

theorem rows_and_columns_exact
    {program : NativeCcsProgram.Program}
    (source : Source program) :
    source.rowCount = program.rows.length + 422465 /\
      source.columnCount =
        NebulaFreshCarrier.alignedPublicWidth +
          (program.columnIds.length - NebulaFreshCarrier.linkWidth) +
          NebulaCombinedLayout.nebulaPrivateWidth := by
  constructor
  · exact source.rowCount_eq
  · rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedCompiler
