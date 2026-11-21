import NightstreamFPrime.Circuit.Execution
import NightstreamFPrime.Export.Generic.MarkedBlocks
import NightstreamFPrime.Export.Generic.RelationCompleteness

/-! The native fills of a step package's witness program (`StepPackage.nativeRanges`,
`SECURITY_MODEL.md`, trust gap 5). Rust fills a block's 16 output locals with the permutation of
its lanes (`fillBlock`) and the outputs of a ring piece with the prior plus the challenge times
the value (`fillRing`), in place of the circuit's batches of the range. When the piece's rows
hold after those batches, the fill gives their values off the block's dead locals
(`agreeOff_block`, `agreeOff_ring`). A batch whose rows are the windows of its marks executes as
the marks' batches (`execute_marks`), so the fills of the marks' blocks give its values
(`agreeOff_batch`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.NativeFill

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.Generic.MarkedBlocks
open NightstreamFPrime.Gadgets.Poseidon2

def fillBlock (block : Block) (env : Env) : Env := fun column =>
  if inside : block.start + 1080 ≤ column ∧ column < block.start + 1096 then
    Permutation.runF Permutation.schedule (block.input env)
      ⟨column - (block.start + 1080), by omega⟩
  else env column

def fillRing (cells start : Nat) (lanes : List Expr) (env : Env) : Env := fun column =>
  if inside : start ≤ column ∧ column < start + ringDegree * cells then
    ringFAdd (priorOf env lanes cells ((column - start) % cells))
      (ringFMul (challengeOf env lanes) (valueOf env lanes cells ((column - start) % cells)))
      ⟨(column - start) / cells, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; omega)⟩
  else env column

theorem lanes_agree {dead : Nat → Prop} {start finish : Nat} {inner : List WitnessBatch}
    (ordered : Witness.Ordered start inner finish) {lanes : List Expr}
    (below : ∀ lane ∈ lanes, lane.VarsBelow start)
    (alive : ∀ lane ∈ lanes, lane.VarsSatisfy fun column => ¬dead column)
    {left right : Env} (agree : AgreeOff dead left right) (index : Nat) :
    (lanes.getD index 0).eval right = (lanes.getD index 0).eval (Witness.execute left inner) := by
  by_cases inside : index < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    have member := List.getElem_mem inside
    exact ((lanes[index]).eval_eq_of_agree_satisfy _ left right (alive _ member) agree).symm.trans
      ((lanes[index]).eval_eq_of_agree_below start _ left (below _ member)
        (Witness.execute_agrees_below ordered left)).symm
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

theorem agreeOff_block {dead : Nat → Prop} {block : Block} (linear : LinearLanes block.lanes)
    {inner : List WitnessBatch}
    (ordered : Witness.Ordered block.start inner (block.start + 1096))
    (deadLocals : ∀ column, block.dead column = true → dead column)
    (below : ∀ lane ∈ block.lanes, lane.VarsBelow block.start)
    (alive : ∀ lane ∈ block.lanes, lane.VarsSatisfy fun column => ¬dead column)
    {left right : Env}
    (rows : ConstraintsHold (Witness.execute left inner)
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())))
    (agree : AgreeOff dead left right) :
    AgreeOff dead (Witness.execute left inner) (fillBlock block right) := by
  have inputs : block.input right = block.input (Witness.execute left inner) := by
    funext lane
    rw [block_input linear, block_input linear]
    exact lanes_agree ordered below alive agree lane.val
  refine Witness.agreeOff_fill ordered (fill := fillBlock block) ?_ ?_ agree
  · intro column outside
    exact dite_eq_right (by omega)
  · intro column lower upper live
    have output : block.start + 1080 ≤ column := by
      by_contra early
      exact live (deadLocals column (by simp [Block.dead]; omega))
    have value := congrFun (block_outputs linear rows) ⟨column - (block.start + 1080), by omega⟩
    rw [← inputs] at value
    unfold fillBlock
    rw [dite_eq_left ⟨output, upper⟩, ← value]
    exact congrArg (Witness.execute left inner)
      (by omega : column = block.start + 1080 + (column - (block.start + 1080)))

/-! ### Marked permutations (SPEC.md section 4.4) -/

def markBatch (mark : PermutationMark) : WitnessBatch :=
  .arithmetic mark.start
    (Permutation.compile mark.start (Permutation.laneState mark.lanes) Permutation.schedule).recipes

def fillMarks (marks : List PermutationMark) (env : Env) : Env :=
  marks.foldl (fun filled mark => fillBlock (blockOf mark) filled) env

theorem markBatch_finish (mark : PermutationMark) :
    (markBatch mark).finish = mark.start + 1096 := by
  simp only [WitnessBatch.finish, markBatch, WitnessBatch.arithmetic_start,
    WitnessBatch.arithmetic_outputLength, Permutation.compile_schedule_recipe_count]

theorem markBatch_ordered {mark : PermutationMark}
    (below : ∀ lane ∈ mark.lanes, lane.VarsBelow mark.start) :
    Witness.Ordered mark.start [markBatch mark] (mark.start + 1096) := by
  refine ⟨le_rfl, ⟨Permutation.compile_schedule_causal _ _ fun lane => ?_, trivial⟩,
    (markBatch_finish mark).le⟩
  unfold Permutation.laneState
  by_cases inside : lane.val < mark.lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact below _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ (by omega)]
    trivial

theorem agreeOff_mark {dead : Nat → Prop} {mark : PermutationMark}
    (lanes : mark.Lanes fun column => ¬dead column)
    (inner : ∀ column, mark.Inner column → dead column) {left right : Env}
    (agree : AgreeOff dead left right) :
    AgreeOff dead (Witness.execute left [markBatch mark]) (fillBlock (blockOf mark) right) := by
  have below : ∀ lane ∈ mark.lanes, lane.VarsBelow mark.start := fun lane member =>
    (Expr.varsSatisfy_lt_iff_varsBelow lane mark.start).mp
      ((lanes.reads lane member).mono lane fun _ inside => inside.1)
  have ordered := markBatch_ordered below
  refine agreeOff_block (block := blockOf mark)
    (List.all_eq_true.mpr fun lane member => linear_of_affine (lanes.affine lane member))
    ordered
    (fun column isDead => by
      simp only [Block.dead, blockOf, Bool.and_eq_true] at isDead
      exact inner column ⟨of_decide_eq_true isDead.1, of_decide_eq_true isDead.2⟩)
    below (fun lane member => (lanes.reads lane member).mono lane fun _ inside => inside.2)
    ?_ agree
  show ConstraintsHold _ (window (PilotData.canonicalConstraints ()) mark)
  rw [PermutationWindow.window_eq_markWindow]
  exact (Witness.execute_computed ordered left (markBatch mark) (List.mem_singleton.mpr rfl)).1

theorem agreeOff_marks {dead : Nat → Prop} (marks : List PermutationMark)
    (lanes : ∀ mark ∈ marks, mark.Lanes fun column => ¬dead column)
    (inner : ∀ mark ∈ marks, ∀ column, mark.Inner column → dead column) {left right : Env}
    (agree : AgreeOff dead left right) :
    AgreeOff dead (Witness.execute left (marks.map markBatch)) (fillMarks marks right) := by
  induction marks generalizing left right with
  | nil => exact agree
  | cons mark rest ih =>
      have head := agreeOff_mark (lanes mark List.mem_cons_self) (inner mark List.mem_cons_self)
        agree
      rw [List.map_cons, ← List.singleton_append, Witness.execute_append]
      exact ih (fun later member => lanes later (List.mem_cons_of_mem _ member))
        (fun later member => inner later (List.mem_cons_of_mem _ member)) head

theorem execute_marks {start : Nat} {recipes : List Expr} {marks : List PermutationMark}
    (rows : recipeConstraints start recipes = marks.flatMap Permutation.markWindow) (env : Env) :
    Witness.execute env [WitnessBatch.arithmetic start recipes] =
      Witness.execute env (marks.map markBatch) := by
  refine Witness.execute_of_rows (marks.map markBatch) env start recipes ?_ ?_
  · intro batch member
    obtain ⟨mark, _, rfl⟩ := List.mem_map.mp member
    exact ⟨rfl, List.ne_nil_of_length_pos (Nat.lt_of_lt_of_eq (by decide : 0 < 1096)
      (Permutation.compile_schedule_recipe_count _ _).symm)⟩
  · rw [rows, List.flatMap_map]
    rfl

/-- The Poseidon2 gadgets place such a batch (`Permutation.markWindow_markOf`,
`Hash.compile_windows`, `Duplex.Formal.compile_windows`). -/
theorem agreeOff_batch {dead : Nat → Prop} {start : Nat} {recipes : List Expr}
    {marks : List PermutationMark}
    (rows : recipeConstraints start recipes = marks.flatMap Permutation.markWindow)
    (lanes : ∀ mark ∈ marks, mark.Lanes fun column => ¬dead column)
    (inner : ∀ mark ∈ marks, ∀ column, mark.Inner column → dead column) {left right : Env}
    (agree : AgreeOff dead left right) :
    AgreeOff dead (Witness.execute left [WitnessBatch.arithmetic start recipes])
      (fillMarks marks right) := by
  rw [execute_marks rows]
  exact agreeOff_marks marks lanes inner agree

theorem agreeOff_ring {dead : Nat → Prop} (canonical : List Expr) {cells start : Nat}
    {lanes : List Expr} {inner : List WitnessBatch}
    (ordered : Witness.Ordered start inner (start + ringDegree * cells))
    (below : ∀ lane ∈ lanes, lane.VarsBelow start)
    (alive : ∀ lane ∈ lanes, lane.VarsSatisfy fun column => ¬dead column)
    {left right : Env}
    (rows : ConstraintsHold (Witness.execute left inner)
      ((Piece.ringProduct cells start lanes).constraints canonical))
    (agree : AgreeOff dead left right) :
    AgreeOff dead (Witness.execute left inner) (fillRing cells start lanes right) := by
  have same := lanes_agree ordered below alive agree
  refine Witness.agreeOff_fill ordered (fill := fillRing cells start lanes) ?_ ?_ agree
  · intro column outside
    exact dite_eq_right (by omega)
  · intro column lower upper _
    have positive : 0 < cells := Nat.pos_of_ne_zero fun zero => by
      rw [zero, Nat.mul_zero] at upper
      omega
    have value := congrFun (output_of_constraints canonical rows
      (Nat.mod_lt (column - start) positive))
      ⟨(column - start) / cells, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; omega)⟩
    have here : outputColumn cells start
        ⟨(column - start) / cells, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; omega)⟩
        ((column - start) % cells) = column := by
      show start + ((column - start) / cells * cells + (column - start) % cells) = column
      rw [Nat.div_add_mod']
      omega
    unfold fillRing
    rw [dite_eq_left ⟨lower, upper⟩]
    have prior : priorOf right lanes cells ((column - start) % cells) =
        priorOf (Witness.execute left inner) lanes cells ((column - start) % cells) :=
      funext fun lane => same _
    have challenge : challengeOf right lanes = challengeOf (Witness.execute left inner) lanes :=
      funext fun lane => same _
    have source : valueOf right lanes cells ((column - start) % cells) =
        valueOf (Witness.execute left inner) lanes cells ((column - start) % cells) :=
      funext fun lane => same _
    rw [prior, challenge, source, ← value]
    exact congrArg (Witness.execute left inner) here.symm

end NightstreamFPrime.Export.Generic.NativeFill
