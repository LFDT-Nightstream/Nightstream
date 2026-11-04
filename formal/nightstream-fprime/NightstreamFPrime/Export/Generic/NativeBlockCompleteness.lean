import NightstreamFPrime.Export.Generic.NativeBlock
import NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
import NightstreamFPrime.Export.Generic.PieceRowsCompleteness

/-! Completeness of native Poseidon2 blocks at a slot layout (CW-8.3), the converse of
`NativeBlock.blocks_sound`. The prover fills each block's 150 S-box columns from `sboxStart` with
the block's template locals (`sboxFill`) and encodes the result in the slots
(`SlotLayoutCompleteness.encode`). When the rows before the binding rows and every block's
template constraints hold in the honest environment and read below the S-box columns, the S-box
rows are zero and the rows and the binding rows hold (`blocks_rows_complete`). Only the columns
that the relation reads need a slot (`ReadsCovered`): the rows' columns, the lanes, the output
locals and the S-box columns. The template constraints read only the lanes and the block's locals
(`templates_of_agree`), so a column that no row reads and a dead local need none. -/

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

/-! ### The columns that the relation reads -/

/-- `column` is read by a row of `rows`, by a lane of a block (its S-box rows read the lanes), by
a block's output locals (its binding rows read them), or is an S-box column, which the S-box rows
read. -/
def Reads (rows : List R1CS.Row) (blocks : List Block) (sboxStart column : Nat) : Prop :=
  (∃ row ∈ rows, column ∈ rowColumns row) ∨
    (∃ block ∈ blocks, (∃ lane ∈ block.lanes, column ∈ exprColumns lane) ∨
      (block.start + 1080 ≤ column ∧ column < block.start + 1096)) ∨
    (sboxStart ≤ column ∧ column < sboxStart + blocks.length * 150)

/-- Each column below the last S-box column that the relation reads is public or has a slot. -/
def ReadsCovered {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (rows : List R1CS.Row) (blocks : List Block) (sboxStart : Nat) : Prop :=
  ∀ column, column < sboxStart + blocks.length * 150 → Reads rows blocks sboxStart column →
    Slotted slots column

theorem ReadsCovered.mono {width : Nat} {slots others : List (LowNormAssignment.Slot width)}
    (included : ∀ slot ∈ slots, slot ∈ others) {rows : List R1CS.Row} {blocks : List Block}
    {sboxStart : Nat} (covered : ReadsCovered slots rows blocks sboxStart) :
    ReadsCovered others rows blocks sboxStart :=
  fun column high reads => (covered column high reads).mono included

/-- A row that reads below `bound` reads only columns below `bound` that `allowed` holds for, when
`allowed` holds for each of its columns below `bound`. -/
theorem row_scope {row : R1CS.Row} {allowed : Nat → Prop} {bound : Nat}
    (below : row.VarsBelow bound) (each : ∀ column ∈ rowColumns row, column < bound → allowed column) :
    row.VarsSatisfy fun column => column < bound ∧ allowed column := by
  have columns := rowColumns_varsSatisfy row
  exact ⟨fun term member => ⟨below.1 term member, each _ (columns.1 term member) (below.1 term member)⟩,
    fun term member => ⟨below.2.1 term member,
      each _ (columns.2.1 term member) (below.2.1 term member)⟩,
    fun term member => ⟨below.2.2 term member,
      each _ (columns.2.2 term member) (below.2.2 term member)⟩⟩

/-! ### The variables of the template constraints -/

theorem subst_varsSatisfy {allowed : Nat → Prop} (wires : Nat → Expr) {bound : Nat}
    (wired : ∀ column, column < bound → (wires column).VarsSatisfy allowed) :
    ∀ expression : Expr, expression.VarsBelow bound → (subst wires expression).VarsSatisfy allowed
  | .var column, below => wired column below
  | .const _, _ => trivial
  | .add left right, below =>
      ⟨subst_varsSatisfy wires wired left below.1, subst_varsSatisfy wires wired right below.2⟩
  | .mul left right, below =>
      ⟨subst_varsSatisfy wires wired left below.1, subst_varsSatisfy wires wired right below.2⟩

theorem varsSatisfy_of_below {allowed : Nat → Prop} {bound : Nat} :
    ∀ expression : Expr, expression.VarsSatisfy (fun column => column < bound → allowed column) →
      expression.VarsBelow bound → expression.VarsSatisfy allowed
  | .var _, scope, below => scope below
  | .const _, _, _ => trivial
  | .add left right, scope, below =>
      ⟨varsSatisfy_of_below left scope.1 below.1, varsSatisfy_of_below right scope.2 below.2⟩
  | .mul left right, scope, below =>
      ⟨varsSatisfy_of_below left scope.1 below.1, varsSatisfy_of_below right scope.2 below.2⟩

theorem recipeConstraints_varsBelow : ∀ (start : Nat) (recipes : List Expr),
    RecipesCausal start recipes →
      ∀ expression ∈ recipeConstraints start recipes, expression.VarsBelow (start + recipes.length)
  | _, [], _ => by simp [recipeConstraints]
  | start, recipe :: rest, causal => by
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · exact varsBelow_sub (show start < start + (rest.length + 1) by omega)
          (Expr.VarsBelow.mono _ causal.1 (by omega))
      · exact Expr.VarsBelow.mono _ (recipeConstraints_varsBelow (start + 1) rest causal.2
          expression member) (by simp only [List.length_cons]; omega)

/-- The canonical Poseidon2 constraints read the 16 inputs and the 1,096 locals. -/
theorem canonical_varsBelow :
    ∀ expression ∈ PilotData.canonicalConstraints (), expression.VarsBelow 1112 := by
  intro expression member
  have causal := Permutation.compile_schedule_causal 16 PilotData.canonicalState
    fun lane => show lane.val < 16 from lane.isLt
  have below := recipeConstraints_varsBelow 16 _ causal expression member
  rwa [Permutation.compile_schedule_recipe_count] at below

/-- A block's template constraints that read below `bound` hold wherever they held, after a change
off its lanes' columns below `bound` and off its 1,096 locals. -/
theorem templates_of_agree {block : Block} {bound : Nat} {before after : Env}
    (below : ∀ expression ∈ (Piece.permutation block.start block.lanes).constraints
      (PilotData.canonicalConstraints ()), expression.VarsBelow bound)
    (lanes : ∀ lane ∈ block.lanes, ∀ column ∈ exprColumns lane, column < bound →
      after column = before column)
    (locals : ∀ column, block.start ≤ column → column < block.start + 1096 →
      after column = before column)
    (holds : ConstraintsHold before ((Piece.permutation block.start block.lanes).constraints
      (PilotData.canonicalConstraints ()))) :
    ConstraintsHold after ((Piece.permutation block.start block.lanes).constraints
      (PilotData.canonicalConstraints ())) := by
  intro expression member
  have scope := below expression member
  obtain ⟨pattern, patternMember, rfl⟩ := List.mem_map.mp member
  have wired : ∀ column, column < 1112 →
      (laneSubst 16 block.start block.lanes column).VarsSatisfy
        fun column => column < bound → after column = before column := by
    intro column inside
    unfold laneSubst
    split
    · by_cases inLanes : column < block.lanes.length
      · rw [List.getD_eq_getElem _ _ inLanes]
        exact (exprColumns_varsSatisfy _).mono _ fun source sourceMember sourceBelow =>
          lanes _ (List.getElem_mem inLanes) source sourceMember sourceBelow
      · rw [List.getD_eq_default _ _ (by omega)]
        trivial
    · exact fun _ => locals _ (by omega) (by omega)
  have support := varsSatisfy_of_below _
    (subst_varsSatisfy _ wired pattern (canonical_varsBelow pattern patternMember)) scope
  rw [Expr.eval_eq_of_agree_satisfy _ _ after before support fun _ agree => agree]
  exact holds _ member

/-! ### Completeness -/

/-- The prover's assignment: the S-box columns filled, then encoded in the slots. -/
def blockAssignment {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (blocks : List Block) (sboxStart : Nat) (honest : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  encode slots (sboxFill blocks sboxStart honest)

/-- Completeness of native blocks at a slot layout: the honest environment's rows and template
constraints give an assignment whose S-box rows are zero and whose rows and binding rows hold,
when each column below the last S-box column that the relation reads is public or has a slot. It
decodes to the honest environment on the public columns. -/
theorem blocks_rows_complete {width : Nat} (slots : List (LowNormAssignment.Slot width))
    {blocks : List Block} (frame : Framed blocks) (sboxStart : Nat) {rows : List R1CS.Row}
    (rowsBelow : ∀ row ∈ rows, row.VarsBelow sboxStart)
    (blocksBelow : ∀ block ∈ blocks, ∀ expression ∈
      (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow sboxStart)
    (publicBelow : LogicalPlan.publicCount ≤ sboxStart)
    (sboxFits : sboxStart + blocks.length * 150 ≤ width)
    (covered : ReadsCovered slots rows blocks sboxStart)
    (rowCount_le : blocks.length * 150 ≤ 2 ^ cubeVariables)
    (honest : Env) (one : honest 0 = 1) (rowsHold : R1CS.RowsHold honest rows)
    (templates : ∀ block ∈ blocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()))) :
    (PoseidonSboxFamilyPlan.plan (sboxInterface (layout slots) blocks sboxStart)
        rowCount_le).RowsZero (blockAssignment slots blocks sboxStart honest) ∧
      R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots)
          (blockAssignment slots blocks sboxStart honest))
        (rows ++ bindingRows width blocks sboxStart) ∧
      ∀ column, column < LogicalPlan.publicCount →
        LogicalPlan.sourceEnv (layout slots) (blockAssignment slots blocks sboxStart honest)
          column = honest column := by
  let filled := sboxFill blocks sboxStart honest
  let assignment := blockAssignment slots blocks sboxStart honest
  have inside : LogicalPlan.publicCount ≤ width := by omega
  have low : ∀ column, column < sboxStart → filled column = honest column :=
    fun column below => sboxFill_outside (Or.inl below)
  have decode : ∀ column, Slotted slots column →
      LogicalPlan.sourceEnv (layout slots) assignment column = filled column :=
    fun column slotted => sourceEnv_encode slots filled (slotted.lt_width inside) slotted
  have read : ∀ column, column < sboxStart + blocks.length * 150 →
      Reads rows blocks sboxStart column →
        LogicalPlan.sourceEnv (layout slots) assignment column = filled column :=
    fun column high reads => decode column (covered column high reads)
  have publicEq : ∀ column, column < LogicalPlan.publicCount →
      LogicalPlan.sourceEnv (layout slots) assignment column = honest column :=
    fun column isPublic => (decode column (Or.inl isPublic)).trans (low column (by omega))
  have oneEq : assignment (layout slots).one = 1 := by
    change encode slots filled (layout slots).one = 1
    rw [encode_one, low 0 (lt_of_lt_of_le (by decide) publicBelow)]
    exact one
  let completed : Env := fun column =>
    if deadIn blocks column then filled column
    else LogicalPlan.sourceEnv (layout slots) assignment column
  have live : ∀ column, deadIn blocks column = false →
      LogicalPlan.sourceEnv (layout slots) assignment column = completed column := by
    intro column dead
    simp only [completed, dead, Bool.false_eq_true, ↓reduceIte]
  have localsAgree : ∀ block ∈ blocks, ∀ column, block.start ≤ column →
      column < block.start + 1096 → completed column = filled column := by
    intro block member column lowColumn highColumn
    cases dead : deadIn blocks column with
    | true => simp only [completed, dead, ↓reduceIte]
    | false =>
        simp only [completed, dead, Bool.false_eq_true, ↓reduceIte]
        have notOwn : block.dead column = false := by
          simp only [deadIn, List.any_eq_false] at dead
          simpa using dead block member
        simp only [Block.dead, Bool.and_eq_false_iff, decide_eq_false_iff_not] at notOwn
        have start := template_start_below (blocksBelow block member)
        exact read column (by omega)
          (Or.inr (Or.inl ⟨block, member, Or.inr ⟨by omega, highColumn⟩⟩))
  have completedTemplates : ∀ block ∈ blocks, ConstraintsHold completed
      ((Piece.permutation block.start block.lanes).constraints
        (PilotData.canonicalConstraints ())) := by
    intro block member
    refine templates_of_agree (blocksBelow block member) ?_ (localsAgree block member)
      (constraintsHold_of_agree_below honest filled _ sboxStart (blocksBelow block member) low
        (templates block member))
    intro lane laneMember column columnMember below
    cases dead : deadIn blocks column with
    | true => simp only [completed, dead, ↓reduceIte]
    | false =>
        simp only [completed, dead, Bool.false_eq_true, ↓reduceIte]
        exact read column (by omega)
          (Or.inr (Or.inl ⟨block, member, Or.inl ⟨lane, laneMember, columnMember⟩⟩))
  have sboxes : ∀ (index : Fin blocks.length) (row : Fin PoseidonRetainedSlots.rows.length),
      LogicalPlan.sourceEnv (layout slots) assignment (sboxColumn sboxStart index.val row.val) =
        completed (sboxLocal (blocks.get index) row) := by
    intro index row
    have bounds := sboxColumn_bounds sboxStart index row
    have member := List.get_mem blocks index
    have start := template_start_below (blocksBelow _ member)
    have localBound := (PoseidonRetainedSlots.localOutput row).isLt
    change (PoseidonRetainedSlots.localOutput row).val < 1096 at localBound
    have localBelow : sboxLocal (blocks.get index) row < sboxStart := by
      unfold sboxLocal
      omega
    rw [read _ bounds.2 (Or.inr (Or.inr bounds)),
      localsAgree _ member _ (by unfold sboxLocal; omega) (by unfold sboxLocal; omega),
      low _ localBelow]
    exact sboxFill_column blocks sboxStart honest index row
  obtain ⟨sboxRows, binding⟩ := blocks_complete frame sboxStart (layout slots) assignment oneEq
    rowCount_le completed completedTemplates live sboxes
  have rowsAt : R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots) assignment) rows :=
    R1CS.rowsHold_of_agree rows (fun column => column < sboxStart ∧ Slotted slots column) honest _
      (fun row member => row_scope (rowsBelow row member) fun column columnMember below =>
        covered column (by omega) (Or.inl ⟨row, member, columnMember⟩))
      (fun column allowed => (decode column allowed.2).trans (low column allowed.1)) rowsHold
  exact ⟨sboxRows, (R1CS.rowsHold_append _ _ _).mpr ⟨rowsAt, binding⟩, publicEq⟩

end NightstreamFPrime.Export.Generic.NativeBlockCompleteness
