import NightstreamFPrime.Export.Stage1.RunningTransitionReducedEncoding
import NightstreamFPrime.Layout.MatrixProgram.PlanBridge

/-! The reduced transition as a composable production matrix plan.
Every declared row decodes; the executable default is unreachable. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedPlan

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionReducedMatrixProgram RunningTransitionReducedMatrixSemantics

private theorem program_decodes {logicalWidth : Nat} (blocks : List MatrixProgram.Block)
    (sourceRow : Nat → Option R1CS.Row)
    (each : ∀ block ∈ blocks, ∀ row : Fin block.rowCount,
      ∃ forms, block.row? logicalWidth sourceRow row.val = some forms) :
    ∀ row : Fin (MatrixProgram.Program.mk blocks).rowCount,
      ∃ forms, (MatrixProgram.Program.mk blocks).row? logicalWidth sourceRow row.val = some forms := by
  induction blocks with
  | nil => intro row; exact Fin.elim0 row
  | cons block rest ih =>
    intro row
    by_cases first : row.val < block.rowCount
    · rw [MatrixProgram.Program.cons_first_row? block rest logicalWidth sourceRow row.val first]
      exact each block List.mem_cons_self ⟨row.val, first⟩
    · have count : (MatrixProgram.Program.mk (block :: rest)).rowCount =
          block.rowCount + (MatrixProgram.Program.mk rest).rowCount := by
        simp only [MatrixProgram.Program.rowCount, List.map_cons, List.sum_cons]
      have bound : row.val - block.rowCount < (MatrixProgram.Program.mk rest).rowCount := by
        have full := row.isLt
        omega
      rw [MatrixProgram.Program.cons_rest_row? block rest logicalWidth sourceRow row.val (by omega)]
      exact ih (fun next member => each next (List.mem_cons_of_mem _ member)) ⟨_, bound⟩

private theorem grid_decodes {logicalWidth : Nat} (block : MultiplicationGrid.Block)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat) (forms : OrdinaryRow.Forms logicalWidth)
    (loaded : block.row? logicalWidth ordinal = some forms) :
    ∃ row, (MatrixProgram.Block.multiplicationGrid block).row? logicalWidth sourceRow ordinal = some row := by
  exact ⟨forms.meaningfulForm, by rw [MatrixProgram.Block.row?, loaded]; rfl⟩

theorem row_decodes {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin (matrixProgram program oneColumn.val).rowCount) :
    ∃ forms, (matrixProgram program oneColumn.val).row? logicalWidth sourceRow row.val = some forms := by
  apply program_decodes (matrixProgram program oneColumn.val).blocks sourceRow ?_ row
  intro block member index
  simp only [matrixProgram, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  · have zero : index.val = 0 := by have bound := index.isLt; change index.val < 1 at bound; omega
    rw [zero]
    exact grid_decodes _ sourceRow 0 _ (flagGrid_row geometry oneColumn)
  · have zero : index.val = 0 := by have bound := index.isLt; change index.val < 1 at bound; omega
    rw [zero]
    exact grid_decodes _ sourceRow 0 _ (bindingGrid_row geometry oneColumn)
  · have zero : index.val = 0 := by have bound := index.isLt; change index.val < 1 at bound; omega
    rw [zero]
    exact grid_decodes _ sourceRow 0 _ (pointHeaderGrid_row geometry oneColumn)
  · have bound : index.val < 56 := index.isLt
    let coordinate : Fin productionShape.cubeVariables := ⟨index.val / 2, by change index.val / 2 < 28; omega⟩
    let part : Fin 2 := ⟨index.val % 2, Nat.mod_lt _ (by decide)⟩
    have address : coordinate.val * 2 + part.val = index.val := by dsimp only [coordinate, part]; omega
    rw [← address]
    exact grid_decodes _ sourceRow _ _ (pointGrid_row geometry oneColumn coordinate part)
  · have count : (MatrixProgram.Block.multiplicationGrid
        (groupsGrid program oneColumn.val)).rowCount = 49296 := by
      simp only [MatrixProgram.Block.rowCount, MultiplicationGrid.Block.rowCount,
        groupsGrid, MultiplicationGrid.Shape.rowCount, Nat.one_mul]
      rfl
    have bound : index.val < 49296 := lt_of_lt_of_eq index.isLt count
    let source : Fin productionShape.runningCount := ⟨index.val / 3081, by change index.val / 3081 < 16; omega⟩
    let minor : Fin PiCCSInputs.runningGroupWords := ⟨index.val % 3081, by change index.val % 3081 < 3081; omega⟩
    have address : source.val * PiCCSInputs.runningGroupWords + minor.val = index.val := by
      change index.val / 3081 * 3081 + index.val % 3081 = index.val
      omega
    obtain ⟨forms, loaded, _⟩ := RunningTransitionReducedGroups.row_preserves relation geometry
      oneColumn (fun _ => 1) rfl source minor
    rw [← address]
    exact grid_decodes _ sourceRow _ forms loaded
  · exact grid_decodes _ sourceRow index.val _ (stateGrid_row geometry oneColumn index)

def plan {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : ProductionRelation.Plan logicalWidth where
  rowCount := (matrixProgram program (RunningTransitionRetainedGeometry.oneColumn geometry).val).rowCount
  rowCount_le := by rw [row_count]; norm_num [Lifecycle.cubeVariables]
  forms := fun row => ((matrixProgram program
    (RunningTransitionRetainedGeometry.oneColumn geometry).val).row?
      logicalWidth (fun _ => none) row.val).getD (fun _ => .empty)

@[simp] theorem plan_rowCount {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : (plan geometry).rowCount = 49359 :=
  row_count program _

theorem matrixProgram_row? {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (sourceRow : Nat → Option R1CS.Row)
    (row : Fin (plan geometry).rowCount) :
    (matrixProgram program (RunningTransitionRetainedGeometry.oneColumn geometry).val).row?
      logicalWidth sourceRow row.val = some ((plan geometry).forms row) := by
  obtain ⟨forms, loaded⟩ := row_decodes relation geometry
    (RunningTransitionRetainedGeometry.oneColumn geometry) (fun _ => none) row
  rw [sourceRow_independent program _ logicalWidth row.val sourceRow (fun _ => none), loaded]
  change some forms = some (((matrixProgram program
    (RunningTransitionRetainedGeometry.oneColumn geometry).val).row?
      logicalWidth (fun _ => none) row.val).getD (fun _ => .empty))
  rw [loaded]
  rfl

theorem rowsZero_iff_accepts {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (sourceRow : Nat → Option R1CS.Row)
    (assignment : Assignment F logicalWidth) :
    (plan geometry).RowsZero assignment ↔
      RunningTransitionReducedMatrixComplete.Accepts
        (matrixProgram program (RunningTransitionRetainedGeometry.oneColumn geometry).val)
        sourceRow assignment := by
  unfold ProductionRelation.Plan.RowsZero RunningTransitionReducedMatrixComplete.Accepts
  apply forall_congr'
  intro row
  rw [ProductionRelation.Plan.rowImage_toVertex, matrixProgram_row? relation geometry sourceRow row]
  change RunningTransitionReducedMatrixComplete.rowResidual ((plan geometry).forms row) assignment = 0 ↔
    ∃ forms, some ((plan geometry).forms row) = some forms ∧
      RunningTransitionReducedMatrixComplete.rowResidual forms assignment = 0
  constructor
  · intro zero; exact ⟨_, rfl, zero⟩
  · rintro ⟨forms, same, zero⟩
    cases Option.some.inj same
    exact zero

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedPlan
