import NightstreamFPrime.Export.Generic.NativeBlock
import NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
import NightstreamFPrime.Export.Generic.PieceRowsCompleteness

/-! Completeness of native Poseidon2 blocks at a slot layout (CW-8.3), the converse of
`NativeBlock.blocks_sound`. The prover fills each block's 150 S-box columns from `sboxStart` with
the block's template locals (`sboxFill`) and encodes the result in the slots
(`SlotLayoutCompleteness.encode`). When the rows before the binding rows and every block's
template constraints hold in the honest environment, read below the S-box columns, and the rows
read no dead local, the S-box rows are zero and the rows and the binding rows hold
(`blocks_rows_complete`). Every live column from the public columns to the last S-box column needs
a slot; the dead locals need none. -/

namespace NightstreamFPrime.Export.Generic.NativeBlockCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.NativeBlock
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2

/-! ### Template locals -/

theorem recipeConstraints_mem :
    ∀ (start : Nat) (recipes : List Expr) (index : Nat) (inside : index < recipes.length),
      Expr.var (start + index) - recipes[index] ∈ recipeConstraints start recipes
  | _, [], _, inside => absurd inside (by simp)
  | start, recipe :: rest, 0, _ => List.mem_cons_self
  | start, recipe :: rest, index + 1, inside => by
      have tail := recipeConstraints_mem (start + 1) rest index (by simpa using inside)
      rw [show start + 1 + index = start + (index + 1) by omega] at tail
      exact List.mem_cons_of_mem _ tail

/-- A block whose template constraints read below `bound` has its 1,096 locals below `bound`:
the last template constraint defines local 1,095. -/
theorem template_start_below {start bound : Nat} {lanes : List Expr}
    (below : ∀ expression ∈ (Piece.permutation start lanes).constraints
      (PilotData.canonicalConstraints ()), expression.VarsBelow bound) :
    start + 1096 ≤ bound := by
  have length : (PilotData.canonicalRecipes ()).length = 1096 :=
    Permutation.compile_schedule_recipe_count 16 PilotData.canonicalState
  have member := recipeConstraints_mem 16 (PilotData.canonicalRecipes ()) 1095
    (by rw [length]; decide)
  have scope := PieceRowsCompleteness.subst_varsBelow (laneSubst 16 start lanes) bound _
    (below _ (List.mem_map_of_mem member))
  have last : (laneSubst 16 start lanes (16 + 1095)).VarsBelow bound := scope.1
  rw [laneSubst_local] at last
  change start + 1095 < bound at last
  omega

theorem dead_below {blocks : List Block} {bound : Nat}
    (below : ∀ block ∈ blocks, ∀ expression ∈ (Piece.permutation block.start block.lanes).constraints
      (PilotData.canonicalConstraints ()), expression.VarsBelow bound)
    {column : Nat} (dead : deadIn blocks column = true) : column < bound := by
  simp only [deadIn, List.any_eq_true, Block.dead, Bool.and_eq_true, decide_eq_true_eq] at dead
  obtain ⟨block, member, _, high⟩ := dead
  have := template_start_below (below block member)
  omega

/-! ### The S-box columns -/

/-- The honest S-box outputs: column `sboxColumn sboxStart index row` holds the template local
`sboxLocal (blocks.get index) row` of `env`. -/
def sboxFill (blocks : List Block) (sboxStart : Nat) (env : Env) : Env := fun column =>
  if sboxStart ≤ column ∧ column < sboxStart + blocks.length * 150 then
    if bounds : (column - sboxStart) / 150 < blocks.length ∧
        (column - sboxStart) % 150 < PoseidonRetainedSlots.rows.length then
      env (sboxLocal (blocks.get ⟨_, bounds.1⟩) ⟨_, bounds.2⟩)
    else env column
  else env column

theorem sboxFill_outside {blocks : List Block} {sboxStart : Nat} {env : Env} {column : Nat}
    (outside : column < sboxStart ∨ sboxStart + blocks.length * 150 ≤ column) :
    sboxFill blocks sboxStart env column = env column := by
  unfold sboxFill
  rw [ite_eq_right (by omega)]

theorem sboxColumn_bounds {blocks : List Block} (sboxStart : Nat) (index : Fin blocks.length)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    sboxStart ≤ sboxColumn sboxStart index.val row.val ∧
      sboxColumn sboxStart index.val row.val < sboxStart + blocks.length * 150 := by
  have rowBound : row.val < 150 := by simpa using row.isLt
  have indexBound := index.isLt
  have : index.val * 150 + 150 ≤ blocks.length * 150 := by
    have := Nat.mul_le_mul_right 150 (Nat.succ_le_of_lt indexBound)
    simpa [Nat.succ_mul] using this
  unfold sboxColumn
  omega

theorem sboxFill_column (blocks : List Block) (sboxStart : Nat) (env : Env)
    (index : Fin blocks.length) (row : Fin PoseidonRetainedSlots.rows.length) :
    sboxFill blocks sboxStart env (sboxColumn sboxStart index.val row.val) =
      env (sboxLocal (blocks.get index) row) := by
  have rowBound : row.val < 150 := by simpa using row.isLt
  have offset : sboxColumn sboxStart index.val row.val - sboxStart = index.val * 150 + row.val := by
    unfold sboxColumn
    omega
  have quotientEq : (index.val * 150 + row.val) / 150 = index.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by decide), Nat.div_eq_of_lt rowBound,
      Nat.zero_add]
  have remainderEq : (index.val * 150 + row.val) % 150 = row.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt rowBound]
  unfold sboxFill
  rw [ite_eq_left (sboxColumn_bounds sboxStart index row)]
  simp only [offset, quotientEq, remainderEq, index.isLt, row.isLt, and_self, ↓reduceDIte]

/-! ### Rows that read live columns below a bound -/

theorem combination_live {dead : Nat → Bool} {bound : Nat} {combination : R1CS.LinearCombination}
    (avoid : combinationAvoids dead combination = true) (below : combination.VarsBelow bound) :
    combination.VarsSatisfy fun column => dead column = false ∧ column < bound := by
  intro term member
  exact ⟨by simpa using List.all_eq_true.mp avoid term member, below term member⟩

theorem row_live {dead : Nat → Bool} {bound : Nat} {row : R1CS.Row}
    (avoid : rowAvoids dead row = true) (below : row.VarsBelow bound) :
    row.VarsSatisfy fun column => dead column = false ∧ column < bound := by
  simp only [rowAvoids, Bool.and_eq_true] at avoid
  exact ⟨combination_live avoid.1.1 below.1, combination_live avoid.1.2 below.2.1,
    combination_live avoid.2 below.2.2⟩

/-! ### Completeness -/

/-- The prover's assignment: the S-box columns filled, then encoded in the slots. -/
def blockAssignment {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (blocks : List Block) (sboxStart : Nat) (honest : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  encode slots (sboxFill blocks sboxStart honest)

/-- Completeness of native blocks at a slot layout: the honest environment's rows and template
constraints give an assignment whose S-box rows are zero and whose rows and binding rows hold. It
decodes to the honest environment on the public columns and to the filled environment on every
live column below the last S-box column. -/
theorem blocks_rows_complete {width : Nat} (slots : List (LowNormAssignment.Slot width))
    {blocks : List Block} (frame : Framed blocks) (sboxStart : Nat) {rows : List R1CS.Row}
    (avoid : rows.all (rowAvoids (deadIn blocks)) = true)
    (rowsBelow : ∀ row ∈ rows, row.VarsBelow sboxStart)
    (blocksBelow : ∀ block ∈ blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ sboxStart)
    (sboxFits : sboxStart + blocks.length * 150 ≤ width)
    (covered : ∀ column, LogicalPlan.publicCount ≤ column →
      column < sboxStart + blocks.length * 150 → deadIn blocks column = false →
        ∃ slot ∈ slots, slot.source.val = column)
    (rowCount_le : blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (one : honest 0 = 1) (rowsHold : R1CS.RowsHold honest rows)
    (templates : ∀ block ∈ blocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()))) :
    (PoseidonSboxFamilyPlan.plan (sboxInterface (layout slots) blocks sboxStart)
        rowCount_le).RowsZero (blockAssignment slots blocks sboxStart honest) ∧
      R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots)
          (blockAssignment slots blocks sboxStart honest))
        (rows ++ bindingRows width blocks sboxStart) ∧
      (∀ column, column < LogicalPlan.publicCount →
        LogicalPlan.sourceEnv (layout slots) (blockAssignment slots blocks sboxStart honest)
          column = honest column) ∧
      ∀ column, column < sboxStart + blocks.length * 150 → deadIn blocks column = false →
        LogicalPlan.sourceEnv (layout slots) (blockAssignment slots blocks sboxStart honest)
          column = sboxFill blocks sboxStart honest column := by
  let filled := sboxFill blocks sboxStart honest
  let assignment := blockAssignment slots blocks sboxStart honest
  have low : ∀ column, column < sboxStart → filled column = honest column :=
    fun column below => sboxFill_outside (Or.inl below)
  have decode : ∀ column, column < sboxStart + blocks.length * 150 →
      deadIn blocks column = false →
        LogicalPlan.sourceEnv (layout slots) assignment column = filled column := by
    intro column high live
    apply sourceEnv_encode slots filled (by omega)
    by_cases isPublic : column < LogicalPlan.publicCount
    · exact Or.inl isPublic
    · exact Or.inr (covered column (Nat.le_of_not_gt isPublic) high live)
  have publicEq : ∀ column, column < LogicalPlan.publicCount →
      LogicalPlan.sourceEnv (layout slots) assignment column = honest column := by
    intro column isPublic
    rw [show LogicalPlan.sourceEnv (layout slots) assignment column = filled column from
      sourceEnv_encode_public slots filled (by omega) isPublic]
    exact low column (by omega)
  have oneEq : assignment (layout slots).one = 1 := by
    change encode slots filled (layout slots).one = 1
    rw [encode_one, low 0 (lt_of_lt_of_le (by decide) publicBelow)]
    exact one
  let completed : Env := fun column =>
    if deadIn blocks column then honest column
    else LogicalPlan.sourceEnv (layout slots) assignment column
  have completedLow : ∀ column, column < sboxStart → completed column = honest column := by
    intro column below
    cases dead : deadIn blocks column with
    | true => simp only [completed, dead, ↓reduceIte]
    | false =>
        simp only [completed, dead, Bool.false_eq_true, ↓reduceIte]
        rw [decode column (by omega) dead]
        exact low column below
  have completedTemplates : ∀ block ∈ blocks, ConstraintsHold completed
      ((Piece.permutation block.start block.lanes).constraints
        (PilotData.canonicalConstraints ())) :=
    fun block member => constraintsHold_of_agree_below honest completed _ sboxStart
      (blocksBelow block member) completedLow (templates block member)
  have live : ∀ column, deadIn blocks column = false →
      LogicalPlan.sourceEnv (layout slots) assignment column = completed column := by
    intro column dead
    simp only [completed, dead, Bool.false_eq_true, ↓reduceIte]
  have sboxes : ∀ (index : Fin blocks.length) (row : Fin PoseidonRetainedSlots.rows.length),
      LogicalPlan.sourceEnv (layout slots) assignment (sboxColumn sboxStart index.val row.val) =
        completed (sboxLocal (blocks.get index) row) := by
    intro index row
    have bounds := sboxColumn_bounds sboxStart index row
    have notDead : deadIn blocks (sboxColumn sboxStart index.val row.val) = false := by
      by_contra dead
      have := dead_below blocksBelow (Bool.eq_true_of_not_eq_false dead)
      omega
    have localBelow : sboxLocal (blocks.get index) row < sboxStart := by
      have start := template_start_below (blocksBelow _ (List.get_mem blocks index))
      have localBound := (PoseidonRetainedSlots.localOutput row).isLt
      change (PoseidonRetainedSlots.localOutput row).val < 1096 at localBound
      unfold sboxLocal
      omega
    rw [decode _ bounds.2 notDead, completedLow _ localBelow]
    exact sboxFill_column blocks sboxStart honest index row
  obtain ⟨sboxRows, binding⟩ := blocks_complete frame sboxStart (layout slots) assignment oneEq
    rowCount_le completed completedTemplates live sboxes
  have rowsAt : R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots) assignment) rows :=
    R1CS.rowsHold_of_agree rows (fun column => deadIn blocks column = false ∧ column < sboxStart)
      honest _ (fun row member => row_live (List.all_eq_true.mp avoid row member)
        (rowsBelow row member))
      (fun column allowed => (decode column (by omega) allowed.1).trans
        (low column allowed.2)) rowsHold
  exact ⟨sboxRows, (R1CS.rowsHold_append _ _ _).mpr ⟨rowsAt, binding⟩, publicEq, decode⟩

end NightstreamFPrime.Export.Generic.NativeBlockCompleteness
