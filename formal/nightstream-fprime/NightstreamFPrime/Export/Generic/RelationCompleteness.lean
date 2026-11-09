import NightstreamFPrime.Export.Generic.CircuitRelation
import NightstreamFPrime.Export.Generic.PieceRowsCompleteness
import NightstreamFPrime.Export.Generic.RowReads
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness

/-! Completeness of a circuit's packed relation (CW-10.2), the converse of
`CircuitRelation.rowsZero_implies_pieces`. Take an environment in which every piece's constraints
hold. Set the blocks' dead locals to 0 (`liveEnv`), add the lowering's fresh columns
(`PieceRowsCompleteness.executePieces`), and put each block's 150 S-box outputs from `sboxStart`,
read from the block's locals (`completeEnv`). An assignment whose layout reads this environment
on each column that the relation reads (`Reads`) satisfies the relation's plan
(`rowsZero_of_holds`): the lowered rows by `executePieces_spec`, the S-box rows because the
block's constraints give its honest trace
(`PoseidonSboxSourceCompleteness.equations_of_sourceRows`), and the binding rows because a block's
outputs are the permutation of its inputs. A column that no row reads has no slot
(`StepPackage.runsOf`, SPEC.md section 4.4).

The final program checks a package's columns without the slot layout: the rows of the pieces other
than the blocks, and each block's outputs as the permutation of its inputs. The lowering's fresh
columns, without the dead locals at 0 or the S-box outputs, satisfy these checks
(`checks_complete`). A piece's constraints read only the columns that `pieceBelow` checks
(`piece_varsBelow`), so they still hold there. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.RelationCompleteness

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
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### Blocks -/

theorem permutation_mem_of_block : ∀ {pieces : List Piece} {block : Block},
    block ∈ blocksOf pieces → Piece.permutation block.start block.lanes ∈ pieces
  | [], _, member => by simp [blocksOf] at member
  | piece :: rest, block, member => by
      cases piece with
      | permutation start lanes =>
          simp only [blocksOf, List.mem_cons] at member
          rcases member with rfl | inRest
          · exact List.mem_cons_self
          · exact List.mem_cons_of_mem _ (permutation_mem_of_block inRest)
      | constraint expression =>
          simp only [blocksOf] at member
          exact List.mem_cons_of_mem _ (permutation_mem_of_block member)
      | ringProduct cells start lanes =>
          simp only [blocksOf] at member
          exact List.mem_cons_of_mem _ (permutation_mem_of_block member)

/-- The canonical environment of a block's invocation reads its lanes and its locals. -/
theorem laneSubst_eval {start : Nat} {lanes : List Expr} (linear : LinearLanes lanes) (env : Env) :
    (fun column => (laneSubst 16 start lanes column).eval env) =
      Pilot.canonicalInvocationEnv (invocation 0 0 start lanes) env := by
  funext column
  by_cases small : column < 16
  · have input := Pilot.canonicalInvocationEnv_input (invocation 0 0 start lanes) env
      ⟨column, small⟩
    simp only at input
    rw [input]
    simp only [laneSubst, small, ite_true]
    exact lane_eval lanes linear env column
  · obtain ⟨index, rfl⟩ : ∃ index, column = 16 + index := ⟨column - 16, by omega⟩
    rw [Pilot.canonicalInvocationEnv_local]
    simp [laneSubst, invocation]

/-- A block's constraints give the canonical constraints in its invocation's environment. -/
theorem block_canonical {start : Nat} {lanes : List Expr} (linear : LinearLanes lanes) {env : Env}
    (holds : ConstraintsHold env
      ((Piece.permutation start lanes).constraints (PilotData.canonicalConstraints ()))) :
    ConstraintsHold (Pilot.canonicalInvocationEnv (invocation 0 0 start lanes) env)
      (PilotData.canonicalConstraints ()) := by
  intro pattern member
  have substituted := holds _ (List.mem_map.mpr ⟨pattern, member, rfl⟩)
  rw [Lifecycle.Nebula.Expr.eval_subst, laneSubst_eval linear] at substituted
  exact substituted

theorem block_input {block : Block} (linear : LinearLanes block.lanes) (env : Env)
    (lane : Fin 16) :
    block.input env lane = (block.lanes.getD lane.val 0).eval env := by
  have same := congrFun (laneSubst_eval (start := block.start) linear env) lane.val
  rw [Pilot.canonicalInvocationEnv_input] at same
  simp only [laneSubst, lane.isLt, ite_true] at same
  exact same.symm

/-- A block's constraints make its outputs the permutation of its inputs. -/
theorem block_outputs {block : Block} (linear : LinearLanes block.lanes) {env : Env}
    (holds : ConstraintsHold env
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ()))) :
    block.output env = Permutation.runF Permutation.schedule (block.input env) := by
  have invocationHolds := Pilot.canonicalPermutationInvocation_complete _ env
    (block_canonical linear holds)
  have sound := Pilot.canonicalPermutationInvocation_sound _ env invocationHolds
  funext lane
  have atLane := congrFun sound lane
  exact atLane

theorem ofFn_runF (state : Fin 16 → F) :
    List.ofFn (Permutation.runF Permutation.schedule state) = Spec.Poseidon2.permute (List.ofFn state) := by
  rw [Permutation.runF_eq_reference, Permutation.runReference_schedule]

/-- An expression below `bound` that avoids the dead columns reads the same values in two
environments that agree on its live columns below `bound`. -/
theorem eval_live {dead : Nat → Bool} {bound : Nat} {left right : Env} :
    ∀ expression : Expr,
      (∀ column ∈ SlotLayout.exprColumns expression, column < bound → dead column = false →
        left column = right column) →
      expression.VarsBelow bound → avoids dead expression = true →
        expression.eval left = expression.eval right
  | .var column, agree, below, avoid => by
      have live : dead column = false := by simpa [avoids] using avoid
      exact agree column (List.mem_singleton_self column) below live
  | .const _, _, _, _ => rfl
  | .add first second, agree, below, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      simp only [Expr.eval_add,
        eval_live first (fun column member => agree column (List.mem_append_left _ member))
          below.1 avoid.1,
        eval_live second (fun column member => agree column (List.mem_append_right _ member))
          below.2 avoid.2]
  | .mul first second, agree, below, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      simp only [Expr.eval_mul,
        eval_live first (fun column member => agree column (List.mem_append_left _ member))
          below.1 avoid.1,
        eval_live second (fun column member => agree column (List.mem_append_right _ member))
          below.2 avoid.2]

/-! ### The completed environment -/

/-- The column after the last S-box output. The relation's rows read no column at or past it. -/
def slotEnd (pieces : List Piece) (next : Nat) : Nat :=
  sboxStart pieces next + 150 * (blocksOf pieces).length

/-- `env` with every dead local of the blocks set to 0. -/
def liveEnv (pieces : List Piece) (env : Env) : Env :=
  fun column => if deadIn (blocksOf pieces) column then 0 else env column

/-- An honest environment completed for the packed relation: the dead locals at 0, the lowering's
fresh columns from `next`, and each block's S-box outputs from `sboxStart`, read from the block's
locals in `env`. -/
noncomputable def completeEnv (pieces : List Piece) (next : Nat) (env : Env) : Env :=
  fun column =>
    if inside : sboxStart pieces next ≤ column ∧ column < slotEnd pieces next then
      env (((blocksOf pieces).get ⟨(column - sboxStart pieces next) / 150, by
          unfold slotEnd at inside; omega⟩).start +
        (PoseidonRetainedSlots.localOutput ⟨(column - sboxStart pieces next) % 150, by
          rw [PoseidonRetainedSlots.rows_length]; omega⟩).val)
    else
      PieceRowsCompleteness.executePieces (PilotData.canonicalConstraints ()) (liveEnv pieces env)
        (otherPieces pieces) next column

theorem completeEnv_low {pieces : List Piece} {next : Nat} {env : Env} {column : Nat}
    (below : column < sboxStart pieces next) :
    completeEnv pieces next env column =
      PieceRowsCompleteness.executePieces (PilotData.canonicalConstraints ()) (liveEnv pieces env)
        (otherPieces pieces) next column := by
  unfold completeEnv
  rw [dite_eq_right (by omega)]

theorem completeEnv_sbox {pieces : List Piece} {next : Nat} {env : Env}
    (index : Fin (blocksOf pieces).length) (row : Fin PoseidonRetainedSlots.rows.length) :
    completeEnv pieces next env (sboxColumn (sboxStart pieces next) index.val row.val) =
      env (((blocksOf pieces).get index).start + (PoseidonRetainedSlots.localOutput row).val) := by
  have rowBound : row.val < 150 := by simpa using row.isLt
  have indexBound := index.isLt
  have room : index.val * 150 + 150 ≤ (blocksOf pieces).length * 150 := by
    have := Nat.mul_le_mul_right 150 (Nat.succ_le_of_lt indexBound)
    simpa [Nat.succ_mul] using this
  unfold completeEnv
  rw [dite_eq_left (by unfold sboxColumn slotEnd; omega)]
  have offset : sboxColumn (sboxStart pieces next) index.val row.val - sboxStart pieces next =
      index.val * 150 + row.val := by
    unfold sboxColumn
    omega
  have indexEq : (index.val * 150 + row.val) / 150 = index.val := by omega
  have rowEq : (index.val * 150 + row.val) % 150 = row.val := by omega
  simp only [offset, indexEq, rowEq]

/-- The columns that the packed relation reads: a column of a term of a lowered row, a column of a
block's lane, a block's output local, which its binding rows read, and an S-box column. -/
def Reads (pieces : List Piece) (next column : Nat) : Prop :=
  (∃ row ∈ (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1,
      column ∈ SlotLayout.rowColumns row) ∨
    (∃ block ∈ blocksOf pieces, (∃ lane ∈ block.lanes, column ∈ SlotLayout.exprColumns lane) ∨
      (block.start + 1080 ≤ column ∧ column < block.start + 1096)) ∨
    (sboxStart pieces next ≤ column ∧ column < slotEnd pieces next)

/-! ### Completeness -/

variable {publicColumns : Nat} {pieces : List Piece} {next width : Nat}

theorem next_le_sboxStart (pieces : List Piece) (next : Nat) : next ≤ sboxStart pieces next := by
  unfold sboxStart
  rw [lowerPieces_eq]
  exact lowerList_next_le _ _ _

/-- The completed environment keeps `env` on the public columns. -/
theorem completeEnv_public (packed : packable publicColumns pieces = true)
    (publicBelow : publicColumns ≤ next) (env : Env) {column : Nat}
    (inside : column < publicColumns) : completeEnv pieces next env column = env column := by
  rw [completeEnv_low (by have := next_le_sboxStart pieces next; omega),
    executePieces_below _ _ _ _ _ (by omega)]
  simp [liveEnv, packable_public packed inside]

/-- The lowered rows hold in the completed environment, when every piece's constraints hold in
`env`, every piece reads only columns below `next`, and the blocks pack. -/
theorem completeEnv_rows (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid) (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    R1CS.RowsHold (completeEnv pieces next env)
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1 := by
  let canonical := PilotData.canonicalConstraints ()
  have checks := packed
  simp only [packable, Bool.and_eq_true, List.all_eq_true] at checks
  have ready : ∀ piece ∈ otherPieces pieces, Ready canonical next (liveEnv pieces env) piece := by
    intro piece member
    obtain ⟨inPieces, notBlock⟩ := List.mem_filter.mp member
    have notBlock' : isBlock piece = false := by simpa using notBlock
    refine ⟨valid piece inPieces, below piece inPieces, ?_, ?_⟩
    · intro blockStart lanes equal
      subst equal
      simp [isBlock] at notBlock'
    · exact avoidsDead_congr (dead := deadIn (blocksOf pieces)) (before := env)
        (fun column live => by simp [liveEnv, live]) canonical piece notBlock'
        (checks.2 piece inPieces) (holds piece inPieces)
  have startEq : (sboxStart pieces next) = (lowerList canonical (otherPieces pieces) next).2 := by
    simp only [sboxStart, lowerPieces_eq, canonical]
  rw [lowerPieces_eq]
  apply (executePieces_spec canonical (otherPieces pieces) (liveEnv pieces env) next ready le_rfl
    (fun _ _ => rfl)).2
  intro column columnBelow
  rw [← startEq] at columnBelow
  exact completeEnv_low columnBelow

/-- An assignment whose layout reads the completed environment on each column that the relation
reads satisfies the plan of the packed relation, when every piece's constraints hold in `env`,
every piece reads only columns below `next`, and the blocks pack. -/
theorem rowsZero_of_holds (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid) (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    {logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment layout.one = 1)
    (represents : ∀ column, Reads pieces next column →
      LogicalPlan.sourceEnv layout assignment column = completeEnv pieces next env column) :
    (plan publicColumns pieces next width fits layout).RowsZero assignment := by
  let canonical := PilotData.canonicalConstraints ()
  have frame := packable_framed packed
  have nextLe : next ≤ (sboxStart pieces next) := next_le_sboxStart pieces next
  have startLe : (sboxStart pieces next) ≤ slotEnd pieces next := Nat.le_add_right _ _
  have sourceLive : ∀ column, Reads pieces next column → column < next →
      deadIn (blocksOf pieces) column = false →
      (LogicalPlan.sourceEnv layout assignment) column = env column := by
    intro column read columnBelow live
    rw [represents column read, completeEnv_low (by omega),
      executePieces_below _ _ _ _ _ columnBelow]
    simp [liveEnv, live]
  -- Each block reads its lanes and its locals below `next`, and its lanes avoid the dead locals.
  have blockFacts : ∀ index : Fin (blocksOf pieces).length,
      LinearLanes ((blocksOf pieces).get index).lanes ∧
        (∀ lane : Fin 16, (((blocksOf pieces).get index).lanes.getD lane.val 0).eval (LogicalPlan.sourceEnv layout assignment) =
          (((blocksOf pieces).get index).lanes.getD lane.val 0).eval env) ∧
        ((blocksOf pieces).get index).start + 1096 ≤ next ∧
        ConstraintsHold env ((Piece.permutation ((blocksOf pieces).get index).start
          ((blocksOf pieces).get index).lanes).constraints canonical) := by
    intro index
    have member := permutation_mem_of_block (List.get_mem (blocksOf pieces) index)
    have pieceBelowNext := below _ member
    simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at pieceBelowNext
    refine ⟨valid _ member, fun lane => ?_, pieceBelowNext.2, holds _ member⟩
    by_cases inside : lane.val < ((blocksOf pieces).get index).lanes.length
    · rw [List.getD_eq_getElem _ _ inside]
      have laneMember := List.getElem_mem inside
      exact eval_live (dead := fun column => ((blocksOf pieces).get index).covers column || deadIn (blocksOf pieces) column) _
        (fun column columnMember columnBelow live => sourceLive column
          (Or.inr (Or.inl ⟨_, List.get_mem (blocksOf pieces) index,
            Or.inl ⟨_, laneMember, columnMember⟩⟩))
          columnBelow (Bool.or_eq_false_iff.mp live).2)
        ((exprBelow_iff next _).mp (List.all_eq_true.mp pieceBelowNext.1 _ laneMember))
        (frame.lanes _ (List.get_mem (blocksOf pieces) index) _ laneMember)
    · rw [List.getD_eq_default _ _ (by omega)]
      rfl
  have rowBound : sboxStart pieces next + (blocksOf pieces).length * 150 ≤ width :=
    (fits.blocks packed).sboxColumns
  let interface := sboxInterface layout (blocksOf pieces) (sboxStart pieces next)
  have sboxColumnBound : ∀ (index : Fin (blocksOf pieces).length) (row : Fin PoseidonRetainedSlots.rows.length),
      sboxColumn (sboxStart pieces next) index.val row.val < slotEnd pieces next ∧
        sboxColumn (sboxStart pieces next) index.val row.val < width := by
    intro index row
    have rowLt : row.val < 150 := by simpa using row.isLt
    have room : index.val * 150 + 150 ≤ (blocksOf pieces).length * 150 := by
      have := Nat.mul_le_mul_right 150 (Nat.succ_le_of_lt index.isLt)
      simpa [Nat.succ_mul] using this
    unfold sboxColumn slotEnd
    constructor <;> omega
  have sboxValue : ∀ (index : Fin (blocksOf pieces).length) (row : Fin PoseidonRetainedSlots.rows.length),
      (interface.sboxOutput index row).eval assignment =
        env (((blocksOf pieces).get index).start + (PoseidonRetainedSlots.localOutput row).val) := by
    intro index row
    change (layout.form (sboxColumn (sboxStart pieces next) index.val row.val)).eval assignment = _
    rw [LogicalPlan.Layout.form_eval,
      represents _ (Or.inr (Or.inr ⟨by unfold sboxColumn; omega, (sboxColumnBound index row).1⟩))]
    exact completeEnv_sbox index row
  have inputValue : ∀ (index : Fin (blocksOf pieces).length),
      SparseLayer.evalState assignment (interface.input index) =
        ((blocksOf pieces).get index).input env := by
    intro index
    funext lane
    obtain ⟨linear, lanesSame, _, _⟩ := blockFacts index
    change (layout.combination
      (invocationInputCombination ((blocksOf pieces).get index).invocation lane.val).toR1CS).eval assignment = _
    rw [LogicalPlan.Layout.combination_eval layout assignment one]
    change ((blocksOf pieces).get index).input (LogicalPlan.sourceEnv layout assignment) lane = _
    rw [block_input linear, block_input linear, lanesSame]
  have sboxRows : (PoseidonSboxFamilyPlan.plan interface
      (by have := (fits.blocks packed).rowCount; omega)).RowsZero assignment := by
    apply PoseidonSboxFamilyPlan.equations_imply_planRowsZero
    intro index
    obtain ⟨linear, _, _, blockHolds⟩ := blockFacts index
    refine ⟨?_, fun lane => by rw [PoseidonSboxFamilyPlan.invocation_output_eq_trace]⟩
    apply PoseidonSboxSourceCompleteness.equations_of_sourceRows _ assignment
      (Pilot.canonicalInvocationEnv ((blocksOf pieces).get index).invocation env) one
    · funext lane
      have atLane := congrFun (inputValue index) lane
      change SparseLayer.evalState assignment (interface.input index) lane = _
      rw [atLane]
      exact (Pilot.canonicalInvocationEnv_input _ env lane).symm
    · intro row
      change (interface.sboxOutput index row).eval assignment = _
      rw [sboxValue, PoseidonRetainedSlots.output_eq_input_add_local]
      exact (Pilot.canonicalInvocationEnv_local ((blocksOf pieces).get index).invocation env
        (PoseidonRetainedSlots.localOutput row).val).symm
    · exact block_canonical linear blockHolds
  have outputValue : ∀ (index : Fin (blocksOf pieces).length),
      SparseLayer.evalState assignment (PoseidonSboxFamilyPlan.Interface.output interface index) =
        Permutation.runF Permutation.schedule (((blocksOf pieces).get index).input env) := by
    intro index
    have permuted := PoseidonSboxFamilyPlan.planRowsZero_implies_permute _ _ assignment one
      sboxRows index
    rw [inputValue index, ← ofFn_runF] at permuted
    exact List.ofFn_injective permuted
  have binding : R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment) (bindingRows width (blocksOf pieces) (sboxStart pieces next)) := by
    intro row member
    obtain ⟨index, -, laneMember⟩ := List.mem_flatMap.mp member
    obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp laneMember
    obtain ⟨linear, _, startBelow, blockHolds⟩ := blockFacts index
    have finals : SparseLayer.evalState (fun column : Fin width => (LogicalPlan.sourceEnv layout assignment) column.val)
          (fun lane => sboxOutputs (width := width) (blocksOf pieces) (sboxStart pieces next) index
            (PoseidonRetainedSlots.finalRow lane)) =
        SparseLayer.evalState assignment
          (fun lane => interface.sboxOutput index (PoseidonRetainedSlots.finalRow lane)) := by
      funext current
      simp only [SparseLayer.evalState, sboxOutputs,
        formOf_ofVar_eval _ _ (sboxColumnBound index (PoseidonRetainedSlots.finalRow current)).2]
      change (LogicalPlan.sourceEnv layout assignment) _ = (layout.form _).eval assignment
      rw [LogicalPlan.Layout.form_eval]
    have outputs := congrFun (outputValue index) lane
    simp only [PoseidonSboxFamilyPlan.Interface.output, SparseLayer.evalState,
      SparseLayer.eval_external] at outputs
    have localValue : (LogicalPlan.sourceEnv layout assignment) (((blocksOf pieces).get index).start + 1080 + lane.val) =
        Permutation.runF Permutation.schedule (((blocksOf pieces).get index).input env) lane := by
      have live : deadIn (blocksOf pieces) (((blocksOf pieces).get index).start + 1080 + lane.val) = false := by
        have laneLt := lane.isLt
        unfold deadIn
        rw [List.any_eq_false]
        intro other otherMember
        simp only [Block.dead, Bool.and_eq_true, decide_eq_true_eq, not_and, not_lt]
        intro otherStart
        by_cases same : other = (blocksOf pieces).get index
        · subst same
          omega
        · have separated := separated_of_mem frame.separated otherMember
            (List.get_mem (blocksOf pieces) index) same
          unfold Separated at separated
          omega
      rw [sourceLive _ (Or.inr (Or.inl ⟨_, List.get_mem (blocksOf pieces) index,
        Or.inr ⟨by omega, by omega⟩⟩)) (by omega) live]
      have atLane := congrFun (block_outputs linear blockHolds) lane
      exact atLane
    simp only [R1CS.Row.Holds, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
      R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_one,
      R1CS.LinearCombination.eval_zero, mul_one, toCombination_eval]
    rw [localValue, ← outputs]
    change _ + -1 * (SparseLayer.external (fun lane => sboxOutputs (width := width) (blocksOf pieces)
      (sboxStart pieces next) index (PoseidonRetainedSlots.finalRow lane)) lane).eval
        (fun column => LogicalPlan.sourceEnv layout assignment column.val) = 0
    rw [SparseLayer.eval_external, finals, neg_one_mul, add_neg_cancel]
  unfold plan
  rw [dite_eq_left packed, Plan.append_rowsZero_iff]
  refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows layout assignment one).mpr ?_,
    sboxRows⟩
  unfold relationRows
  rw [ite_eq_left packed, R1CS.rowsHold_append]
  refine ⟨?_, binding⟩
  exact R1CS.rowsHold_of_agree _ (Reads pieces next) (completeEnv pieces next env) _
    (fun row member => (RowReads.rowColumns_varsSatisfy row).mono row
      fun column columnMember => Or.inl ⟨row, member, columnMember⟩)
    represents (completeEnv_rows packed valid below env holds)

/-! ### The final program's checks -/

theorem subst_varsBelow {inner outer : Nat} (wires : Nat → Expr)
    (wiresBelow : ∀ column, column < inner → (wires column).VarsBelow outer) :
    ∀ expression : Expr, expression.VarsBelow inner → (subst wires expression).VarsBelow outer
  | .var column, below => wiresBelow column below
  | .const _, _ => trivial
  | .add left right, below => by
      show (subst wires left).VarsBelow outer ∧ (subst wires right).VarsBelow outer
      exact ⟨subst_varsBelow wires wiresBelow left below.1,
        subst_varsBelow wires wiresBelow right below.2⟩
  | .mul left right, below => by
      show (subst wires left).VarsBelow outer ∧ (subst wires right).VarsBelow outer
      exact ⟨subst_varsBelow wires wiresBelow left below.1,
        subst_varsBelow wires wiresBelow right below.2⟩

/-- The canonical permutation reads its 16 inputs and its 1,096 locals. -/
theorem canonical_varsBelow :
    ∀ expression ∈ PilotData.canonicalConstraints (), expression.VarsBelow 1112 := by
  intro expression member
  have scope := recipeConstraints_varsBelow_of_causal 16 (PilotData.canonicalRecipes ())
    (Permutation.compile_schedule_causal 16 PilotData.canonicalState fun lane => lane.isLt)
    expression member
  rw [show (PilotData.canonicalRecipes ()).length = 1096 from
    Permutation.compile_schedule_recipe_count 16 PilotData.canonicalState] at scope
  exact scope

/-- The constraints of a piece that reads only columns below `bound` read only columns below
`bound`. -/
theorem piece_varsBelow {bound : Nat} :
    ∀ piece : Piece, pieceBelow bound piece = true →
      ∀ expression ∈ piece.constraints (PilotData.canonicalConstraints ()),
        expression.VarsBelow bound
  | .constraint expression, below, current, member => by
      simp only [Piece.constraints, List.mem_singleton] at member
      subst member
      exact (exprBelow_iff bound _).mp below
  | .permutation start lanes, below, current, member => by
      simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at below
      simp only [Piece.constraints, List.mem_map] at member
      obtain ⟨template, inTemplate, rfl⟩ := member
      refine subst_varsBelow _ (fun column inside => ?_) template
        (canonical_varsBelow template inTemplate)
      unfold laneSubst
      split
      · exact lanes_below below.1 column
      · show start + (column - 16) < bound
        omega
  | .ringProduct cells start lanes, below, current, member => by
      simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at below
      simp only [Piece.constraints, ringTemplate, List.mem_map, List.mem_ofFn] at member
      obtain ⟨_, ⟨index, rfl⟩, rfl⟩ := member
      refine subst_varsBelow _ (fun column inside => ?_) _ (ringConstraint_varsBelow cells index)
      unfold laneSubst
      split
      · exact lanes_below below.1 column
      · show start + (column - ringInputs cells) < bound
        omega

/-- The constraints that `split` cut read only columns below `bound` when every piece does. -/
theorem split_varsBelow {constraints : List Expr} {bound : Nat}
    (below : (split (PilotData.canonicalConstraints ()) constraints).all (pieceBelow bound) =
      true) :
    ∀ expression ∈ constraints, expression.VarsBelow bound := by
  intro expression member
  rw [← flatMap_split (PilotData.canonicalConstraints ()) constraints, List.mem_flatMap] at member
  obtain ⟨piece, pieceMember, inPiece⟩ := member
  exact piece_varsBelow piece (List.all_eq_true.mp below piece pieceMember) expression inPiece

/-- Completeness of the final program's checks on a package's columns, the converse of
`CircuitRelation.checks_implies_pieces`: when every piece's constraints hold in `env` and read
only columns below `next`, the lowering's fresh columns give an environment that agrees with
`env` below `next`, in which every piece's constraints and the rows of the pieces other than the
blocks hold, and each block's outputs are the permutation of its inputs. -/
theorem checks_complete (pieces : List Piece) (valid : ∀ piece ∈ pieces, piece.Valid)
    (next : Nat) (below : pieces.all (pieceBelow next) = true) (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    ∃ completed : Env,
      (∀ column, column < next → completed column = env column) ∧
      (∀ piece ∈ pieces,
        ConstraintsHold completed (piece.constraints (PilotData.canonicalConstraints ()))) ∧
      R1CS.RowsHold completed
        (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1 ∧
      OutputsHold (blocksOf pieces) completed := by
  let canonical := PilotData.canonicalConstraints ()
  have ready : ∀ piece ∈ otherPieces pieces, Ready canonical next env piece := by
    intro piece member
    obtain ⟨inPieces, notBlock⟩ := List.mem_filter.mp member
    refine ⟨valid piece inPieces, List.all_eq_true.mp below piece inPieces, ?_,
      holds piece inPieces⟩
    intro blockStart lanes equal
    subst equal
    simp [isBlock] at notBlock
  obtain ⟨agree, rows⟩ :=
    executePieces_spec canonical (otherPieces pieces) env next ready le_rfl fun _ _ => rfl
  have kept : ∀ column, column < next →
      executePieces canonical env (otherPieces pieces) next column = env column :=
    fun column inside => agree column (Or.inl inside)
  have piecesHold : ∀ piece ∈ pieces, ConstraintsHold
      (executePieces canonical env (otherPieces pieces) next) (piece.constraints canonical) :=
    fun piece member => constraintsHold_of_agree_below env _ _ next
      (piece_varsBelow piece (List.all_eq_true.mp below piece member)) kept (holds piece member)
  refine ⟨_, kept, piecesHold, ?_, fun block member => ?_⟩
  · rw [lowerPieces_eq]
    exact rows _ fun _ _ => rfl
  · have pieceMember := permutation_mem_of_block member
    exact block_outputs (valid _ pieceMember) (piecesHold _ pieceMember)

end NightstreamFPrime.Export.Generic.RelationCompleteness
