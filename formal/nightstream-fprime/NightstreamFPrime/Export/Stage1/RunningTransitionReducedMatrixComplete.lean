import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixPoint
import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixState
import NightstreamFPrime.Export.Stage1.RunningTransitionReducedGroups

/-!
Compose the six actual compact row families into the complete reduced
running-transition relation. Row decoding and the fixed production polynomial
supply acceptance; no caller supplies row arithmetic or source-value equalities.
The program remains unselected until aggregate layout transport is complete.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixComplete

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionReducedMatrixProgram RunningTransitionReducedMatrixSemantics

/-- Evaluate the fixed 14-matrix polynomial on an actually decoded sparse row. -/
def rowResidual {logicalWidth : Nat} (forms : RowForms logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  evaluatePolynomial baseOps Spec.ProductionRelation.polynomial (fun port =>
    (match Layout.ProductionRelation.meaningfulPort? port with
      | some meaningful => forms meaningful
      | none => SparseForm.empty).eval assignment)

def RowZero {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (decoded : Option (RowForms logicalWidth)) : Prop :=
  ∃ forms, decoded = some forms ∧ rowResidual forms assignment = 0

/-- Acceptance includes successful decoding of every declared row. -/
def Accepts {logicalWidth : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin program.rowCount, RowZero assignment (program.row? logicalWidth sourceRow row.val)

private def BlockAccepts {logicalWidth : Nat} (block : MatrixProgram.Block)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row : Fin block.rowCount, RowZero assignment (block.row? logicalWidth sourceRow row.val)

private theorem accepts_cons {logicalWidth : Nat} (block : MatrixProgram.Block)
    (rest : List MatrixProgram.Block) (sourceRow : Nat → Option R1CS.Row)
    (assignment : Assignment F logicalWidth) :
    Accepts ⟨block :: rest⟩ sourceRow assignment ↔
      BlockAccepts block sourceRow assignment ∧ Accepts ⟨rest⟩ sourceRow assignment := by
  have count : (MatrixProgram.Program.mk (block :: rest)).rowCount =
      block.rowCount + (MatrixProgram.Program.mk rest).rowCount := by
    simp only [MatrixProgram.Program.rowCount, List.map_cons, List.sum_cons]
  constructor
  · intro accepted
    constructor
    · intro row
      have bound : row.val < (MatrixProgram.Program.mk (block :: rest)).rowCount := by
        rw [count]
        omega
      have selected := accepted ⟨row.val, bound⟩
      rw [MatrixProgram.Program.cons_first_row? block rest logicalWidth sourceRow row.val row.isLt]
        at selected
      exact selected
    · intro row
      have bound : block.rowCount + row.val <
          (MatrixProgram.Program.mk (block :: rest)).rowCount := by
        rw [count]
        exact Nat.add_lt_add_left row.isLt _
      have selected := accepted ⟨block.rowCount + row.val, bound⟩
      rw [MatrixProgram.Program.cons_rest_row? block rest logicalWidth sourceRow
        (block.rowCount + row.val) (by omega), Nat.add_sub_cancel_left] at selected
      exact selected
  · rintro ⟨first, tail⟩ row
    by_cases before : row.val < block.rowCount
    · rw [MatrixProgram.Program.cons_first_row? block rest logicalWidth sourceRow row.val before]
      exact first ⟨row.val, before⟩
    · have bound : row.val - block.rowCount < (MatrixProgram.Program.mk rest).rowCount := by
        have full := row.isLt
        omega
      rw [MatrixProgram.Program.cons_rest_row? block rest logicalWidth sourceRow row.val (by omega)]
      exact tail ⟨row.val - block.rowCount, bound⟩

private theorem accepts_nil {logicalWidth : Nat} (sourceRow : Nat → Option R1CS.Row)
    (assignment : Assignment F logicalWidth) : Accepts ⟨[]⟩ sourceRow assignment := by
  intro row
  exact Fin.elim0 row

private theorem ordinary_rowZero {logicalWidth : Nat} (forms : OrdinaryRow.Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    RowZero assignment (some forms.meaningfulForm) ↔ forms.residual assignment = 0 := by
  constructor
  · rintro ⟨loaded, same, zero⟩
    cases Option.some.inj same
    exact zero
  · intro zero
    exact ⟨forms.meaningfulForm, rfl, zero⟩

private theorem grid_row_iff {logicalWidth : Nat} (block : MultiplicationGrid.Block)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth)
    (env : Env) (ordinal : Nat) (row : R1CS.Row) (forms : OrdinaryRow.Forms logicalWidth)
    (loaded : block.row? logicalWidth ordinal = some forms)
    (preserves : forms.Preserves assignment env row) :
    RowZero assignment ((MatrixProgram.Block.multiplicationGrid block).row?
      logicalWidth sourceRow ordinal) ↔ row.Holds env := by
  rw [MatrixProgram.Block.row?, loaded]
  change RowZero assignment (some forms.meaningfulForm) ↔ row.Holds env
  rw [ordinary_rowZero]
  exact forms.residual_zero_iff assignment env row preserves

private theorem singleton_grid_iff {logicalWidth : Nat} (block : MultiplicationGrid.Block)
    (count : block.rowCount = 1) (sourceRow : Nat → Option R1CS.Row)
    (assignment : Assignment F logicalWidth) (env : Env)
    (row : R1CS.Row) (forms : OrdinaryRow.Forms logicalWidth)
    (loaded : block.row? logicalWidth 0 = some forms)
    (preserves : forms.Preserves assignment env row) :
    BlockAccepts (.multiplicationGrid block) sourceRow assignment ↔ row.Holds env := by
  have rowEquivalent := grid_row_iff block sourceRow assignment env 0 row forms loaded preserves
  constructor
  · intro all
    exact rowEquivalent.mp (all ⟨0, by change 0 < block.rowCount; rw [count]; decide⟩)
  · intro holds index
    have bound := index.isLt
    change index.val < block.rowCount at bound
    rw [count] at bound
    have value : index.val = 0 := by omega
    rw [value]
    exact rowEquivalent.mpr holds

private theorem forall_product {major minor : Nat} (predicate : Fin (major * minor) → Prop) :
    (∀ index, predicate index) ↔
      ∀ first : Fin major, ∀ second : Fin minor, predicate (Fin.encodeProd (first, second)) := by
  constructor
  · intro all first second
    exact all _
  · intro all index
    have result := all (Fin.decodeProd index).1 (Fin.decodeProd index).2
    change predicate (Fin.encodeProd (Fin.decodeProd index)) at result
    rw [Fin.encodeProd_decodeProd] at result
    exact result

private theorem point_accepted_iff {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth)
    (one : assignment oneColumn = 1) :
    BlockAccepts (.multiplicationGrid (pointGrid program oneColumn.val)) sourceRow assignment ↔
      ∀ coordinate : Fin productionShape.cubeVariables, ∀ part : Fin 2,
        (RunningTransitionReducedRows.muxRow relationWidth publicFits
          (RunningTransitionWordIndexing.pointIndex coordinate part)).Holds (decodedEnv geometry assignment) := by
  change (∀ index : Fin (productionShape.cubeVariables * 2),
    RowZero assignment ((MatrixProgram.Block.multiplicationGrid (pointGrid program oneColumn.val)).row?
      logicalWidth sourceRow index.val)) ↔ _
  rw [forall_product]
  apply forall_congr'
  intro coordinate
  apply forall_congr'
  intro part
  simpa only [Fin.encodeProd, Fin.mkDivMod, Nat.mul_comm] using grid_row_iff
    (pointGrid program oneColumn.val) sourceRow assignment (decodedEnv geometry assignment)
    (coordinate.val * 2 + part.val) _ (pointForms geometry oneColumn coordinate part)
    (pointGrid_row geometry oneColumn coordinate part)
    (pointForms_preserve relation geometry oneColumn assignment one coordinate part)

private theorem groups_accepted_iff {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth)
    (one : assignment oneColumn = 1) :
    BlockAccepts (.multiplicationGrid (groupsGrid program oneColumn.val)) sourceRow assignment ↔
      ∀ source : Fin productionShape.runningCount, ∀ minor : Fin PiCCSInputs.runningGroupWords,
        (RunningTransitionReducedRows.muxRow relationWidth publicFits
          (RunningTransitionWordIndexing.groupIndex source minor)).Holds (decodedEnv geometry assignment) := by
  have count : (groupsGrid program oneColumn.val).rowCount =
      productionShape.runningCount * PiCCSInputs.runningGroupWords := by
    simp only [groupsGrid, MultiplicationGrid.Block.rowCount,
      MultiplicationGrid.Shape.rowCount, Nat.one_mul]
  unfold BlockAccepts
  change (∀ index : Fin (groupsGrid program oneColumn.val).rowCount,
    RowZero assignment ((MatrixProgram.Block.multiplicationGrid (groupsGrid program oneColumn.val)).row?
      logicalWidth sourceRow index.val)) ↔ _
  rw [count, forall_product]
  apply forall_congr'
  intro source
  apply forall_congr'
  intro minor
  obtain ⟨forms, loaded, preserves⟩ := RunningTransitionReducedGroups.row_preserves
    relation geometry oneColumn assignment one source minor
  simpa only [Fin.encodeProd, Fin.mkDivMod, Nat.mul_comm] using grid_row_iff
    (groupsGrid program oneColumn.val) sourceRow assignment (decodedEnv geometry assignment)
    (source.val * PiCCSInputs.runningGroupWords + minor.val) _ forms loaded preserves

private theorem state_accepted_iff {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth)
    (one : assignment oneColumn = 1) :
    BlockAccepts (.multiplicationGrid (stateGrid program oneColumn.val)) sourceRow assignment ↔
      ∀ index, (RunningTransitionReducedRows.stateRow index).Holds (decodedEnv geometry assignment) := by
  change (∀ index : RunningTransition.StateIndex, RowZero assignment
    ((MatrixProgram.Block.multiplicationGrid (stateGrid program oneColumn.val)).row?
      logicalWidth sourceRow index.val)) ↔ _
  apply forall_congr'
  intro index
  exact grid_row_iff (stateGrid program oneColumn.val) sourceRow assignment
    (decodedEnv geometry assignment) index.val _ (stateForms geometry oneColumn index)
    (stateGrid_row geometry oneColumn index)
    (stateForms_preserve geometry oneColumn assignment one index)

private theorem forall_words (predicate : RunningTransition.WordIndex → Prop) :
    (∀ index, predicate index) ↔ predicate ⟨0, by decide⟩ ∧
      (∀ coordinate : Fin productionShape.cubeVariables, ∀ part : Fin 2,
        predicate (RunningTransitionWordIndexing.pointIndex coordinate part)) ∧
      (∀ source : Fin productionShape.runningCount, ∀ minor : Fin PiCCSInputs.runningGroupWords,
        predicate (RunningTransitionWordIndexing.groupIndex source minor)) := by
  constructor
  · intro all
    exact ⟨all _, fun _ _ => all _, fun _ _ => all _⟩
  · rintro ⟨header, points, groups⟩ index
    have bound : index.val < 49353 := index.isLt
    by_cases zero : index.val = 0
    · have same : index = ⟨0, by decide⟩ := Fin.ext zero
      simpa only [same] using header
    · by_cases point : index.val < 57
      · let coordinate : Fin productionShape.cubeVariables :=
          ⟨(index.val - 1) / 2, by change (index.val - 1) / 2 < 28; omega⟩
        let part : Fin 2 := ⟨(index.val - 1) % 2, Nat.mod_lt _ (by decide)⟩
        have same : RunningTransitionWordIndexing.pointIndex coordinate part = index := by
          apply Fin.ext
          rw [RunningTransitionWordIndexing.pointIndex_val]
          dsimp only [coordinate, part]
          omega
        simpa only [same] using points coordinate part
      · let source : Fin productionShape.runningCount :=
          ⟨(index.val - 57) / 3081, by change (index.val - 57) / 3081 < 16; omega⟩
        let minor : Fin PiCCSInputs.runningGroupWords :=
          ⟨(index.val - 57) % 3081, Nat.mod_lt _ (by decide)⟩
        have same : RunningTransitionWordIndexing.groupIndex source minor = index := by
          apply Fin.ext
          rw [RunningTransitionWordIndexing.groupIndex_val]
          dsimp only [source, minor]
          omega
        simpa only [same] using groups source minor

private theorem rowsHold_cons (env : Env) (row : R1CS.Row) (rest : List R1CS.Row) :
    R1CS.RowsHold env (row :: rest) ↔ row.Holds env ∧ R1CS.RowsHold env rest := by
  constructor
  · intro all
    exact ⟨all _ List.mem_cons_self, fun next member => all next (List.mem_cons_of_mem _ member)⟩
  · rintro ⟨head, tail⟩ next member
    rcases List.mem_cons.mp member with same | member
    · simpa only [same] using head
    · exact tail next member

private theorem rowsHold_ofFn (env : Env) {count : Nat} (rows : Fin count → R1CS.Row) :
    R1CS.RowsHold env (List.ofFn rows) ↔ ∀ index, (rows index).Holds env := by
  constructor
  · intro all index
    exact all _ (List.mem_ofFn.mpr ⟨index, rfl⟩)
  · intro all row member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact all index

/-- The declared Fin 49,359 domain splits into these six ordered families. -/
theorem block_row_counts (program : ApplicationProgram) (oneColumn : Nat) :
    ((matrixProgram program oneColumn).blocks.map MatrixProgram.Block.rowCount) =
      [1, 1, 1, 56, 49296, 4] := by
  simp only [matrixProgram, List.map_cons, List.map_nil, MatrixProgram.Block.rowCount,
    MultiplicationGrid.Block.rowCount, flagGrid, bindingGrid, pointHeaderGrid,
    pointGrid, groupsGrid, stateGrid, MultiplicationGrid.Shape.rowCount,
    Nat.one_mul, Nat.mul_one]
  rfl

/-- Actual compact decoding and the fixed polynomial accept exactly the reduced
source relation. All row-value preservation premises are discharged by owners. -/
theorem accepts_iff_rows {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth)
    (one : assignment oneColumn = 1) :
    Accepts (matrixProgram program oneColumn.val) sourceRow assignment ↔
      R1CS.RowsHold (decodedEnv geometry assignment)
        (RunningTransitionReducedRows.rows relationWidth publicFits) := by
  have flag := singleton_grid_iff (flagGrid program oneColumn.val) rfl sourceRow assignment
    (decodedEnv geometry assignment) RunningTransitionReducedRows.flagRow
    (flagForms geometry oneColumn) (flagGrid_row geometry oneColumn)
    (flagForms_preserve geometry oneColumn assignment one)
  have binding := singleton_grid_iff (bindingGrid program oneColumn.val) rfl sourceRow assignment
    (decodedEnv geometry assignment) RunningTransitionReducedRows.bindingRow
    (bindingForms geometry oneColumn) (bindingGrid_row geometry oneColumn)
    (bindingForms_preserve geometry oneColumn assignment one)
  have header := singleton_grid_iff (pointHeaderGrid program oneColumn.val) rfl sourceRow assignment
    (decodedEnv geometry assignment) _ (pointHeaderForms geometry oneColumn)
    (pointHeaderGrid_row geometry oneColumn)
    (pointHeaderForms_preserve relation geometry oneColumn assignment one)
  have points := point_accepted_iff relation geometry oneColumn sourceRow assignment one
  have groups := groups_accepted_iff relation geometry oneColumn sourceRow assignment one
  have state := state_accepted_iff geometry oneColumn sourceRow assignment one
  rw [matrixProgram, accepts_cons, accepts_cons, accepts_cons, accepts_cons, accepts_cons, accepts_cons]
  rw [flag, binding, header, points, groups, state]
  rw [RunningTransitionReducedRows.rows, rowsHold_cons, rowsHold_cons,
    R1CS.rowsHold_append, rowsHold_ofFn, rowsHold_ofFn, forall_words]
  have empty := accepts_nil sourceRow assignment
  constructor
  · rintro ⟨flag, binding, header, points, groups, state, _⟩
    exact ⟨flag, binding, ⟨header, points, groups⟩, state⟩
  · rintro ⟨flag, binding, ⟨header, points, groups⟩, state⟩
    exact ⟨flag, binding, header, points, groups, state, empty⟩

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixComplete
