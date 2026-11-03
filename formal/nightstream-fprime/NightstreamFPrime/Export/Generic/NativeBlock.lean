import NightstreamFPrime.Export.Generic.CircuitRelation
import NightstreamFPrime.Export.PermutationOutput
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness
import NightstreamFPrime.Spec.GoldilocksPrime

/-! Native Poseidon2 blocks and bit slots (CW-8.3). A Rust-built application can place each
Poseidon2 permutation as an option-B block (`Frame.Block`) instead of its 1,096 template rows: 150
S-box rows (`PoseidonSboxFamilyPlan`) on S-box columns from `sboxStart`, 16 binding rows
(`CircuitRelation.bindingRows`), and no slot for its 1,080 dead locals. The statements hold for any
blocks, rows, layout and assignment, whoever builds the package.

Soundness (`blocks_sound`): when the S-box rows are zero and the other rows and the binding rows
hold, each block's outputs are `Spec.Poseidon2.permute` of its lanes, and an environment that
differs only on the dead locals satisfies the other rows and every block's 1,096 template rows.
It needs the frame (`Frame.Framed`, checked by `Frame.checkFrame`) and other rows that read no
dead local (`rowAvoids`). Completeness (`blocks_complete`): an honest template trace, read off the
dead locals and with each S-box column at its template local, makes the S-box rows zero and the
binding rows hold.

A `bit` slot commits one coordinate. A value 0 or 1 encodes within the norm bound
(`LowNormSlot.encode_norm`, `bitHint_valid`, `bit_valid_of_boolean`), and a slot whose coordinate
passes the norm check decodes to -1, 0 or 1 (`bitSlot_sound`). -/

namespace NightstreamFPrime.Export.Generic.NativeBlock

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
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2

/-! ### Template constraints and template rows -/

/-- The template environment of a block: input `j` reads lane `j`, local `k` reads column
`start + k`. -/
theorem laneEnv_eq (phase rowStart start : Nat) (lanes : List Expr) (linear : LinearLanes lanes)
    (env : Env) :
    (fun column => (laneSubst 16 start lanes column).eval env) =
      Pilot.canonicalInvocationEnv (invocation phase rowStart start lanes) env := by
  funext column
  by_cases small : column < 16
  · have input := Pilot.canonicalInvocationEnv_input
      (invocation phase rowStart start lanes) env ⟨column, small⟩
    simp only at input
    rw [input]
    simp only [laneSubst, small, ite_true]
    exact lane_eval lanes linear env column
  · obtain ⟨index, rfl⟩ : ∃ index, column = 16 + index := ⟨column - 16, by omega⟩
    rw [Pilot.canonicalInvocationEnv_local]
    simp [laneSubst, invocation]

/-- A block's template constraints hold exactly when its 1,096 template rows hold. -/
theorem constraints_iff_invocation (phase rowStart start : Nat) (lanes : List Expr)
    (linear : LinearLanes lanes) (env : Env) :
    ConstraintsHold env
        ((Piece.permutation start lanes).constraints (PilotData.canonicalConstraints ())) ↔
      PermutationInvocationHolds (PilotData.circuitPackage ())
        (invocation phase rowStart start lanes) env := by
  refine ⟨fun holds => ?_, permutation_sound phase rowStart start lanes linear env⟩
  have substituted : ConstraintsHold env ((PilotData.canonicalConstraints ()).map
      (subst (laneSubst 16 start lanes))) := holds
  apply Pilot.canonicalPermutationInvocation_complete
  rw [← laneEnv_eq phase rowStart start lanes linear env]
  intro pattern member
  have value := substituted _ (List.mem_map_of_mem member)
  rwa [Lifecycle.Nebula.Expr.eval_subst] at value

/-! ### Rows that read no dead local -/

/-- `combination` reads no column that `dead` marks. -/
def combinationAvoids (dead : Nat → Bool) (combination : R1CS.LinearCombination) : Bool :=
  combination.terms.all fun term => !dead term.1

/-- `row` reads no column that `dead` marks. -/
def rowAvoids (dead : Nat → Bool) (row : R1CS.Row) : Bool :=
  combinationAvoids dead row.a && combinationAvoids dead row.b && combinationAvoids dead row.c

theorem combination_eval_congr {dead : Nat → Bool} {before after : Env}
    (agree : ∀ column, dead column = false → after column = before column)
    {combination : R1CS.LinearCombination} (avoid : combinationAvoids dead combination = true) :
    combination.eval after = combination.eval before := by
  unfold R1CS.LinearCombination.eval
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro term member
  have live : dead term.1 = false := by
    simpa using List.all_eq_true.mp avoid term member
  rw [agree term.1 live]

/-- Rows that read no dead local hold wherever they held, after a change of dead columns only. -/
theorem rowsHold_congr {dead : Nat → Bool} {before after : Env}
    (agree : ∀ column, dead column = false → after column = before column)
    {rows : List R1CS.Row} (avoid : rows.all (rowAvoids dead) = true)
    (holds : R1CS.RowsHold before rows) : R1CS.RowsHold after rows := by
  intro row member
  have rowAvoid := List.all_eq_true.mp avoid row member
  simp only [rowAvoids, Bool.and_eq_true] at rowAvoid
  have rowHolds : row.Holds before := holds row member
  unfold R1CS.Row.Holds at rowHolds ⊢
  rw [combination_eval_congr agree rowAvoid.1.1, combination_eval_congr agree rowAvoid.1.2,
    combination_eval_congr agree rowAvoid.2]
  exact rowHolds

/-! ### Soundness -/

/-- `CircuitRelation.termsForm_eval` without the width premise: a term at or beyond the width
reads 0 on both sides. -/
theorem termsForm_eval_any {width : Nat} (assignment : Assignment F width) :
    ∀ terms : List (Nat × F),
      (termsForm terms).eval assignment =
        (terms.map fun term => term.2 * identityEnv assignment term.1).sum
  | [] => by simp [termsForm]
  | (column, coefficient) :: rest => by
      by_cases inside : column < width
      · simp only [termsForm, inside, dite_true, SparseForm.add_eval, SparseForm.singleton_eval,
          List.map_cons, List.sum_cons, termsForm_eval_any assignment rest]
        simp [identityEnv, inside]
      · simp only [termsForm, inside, dite_false, List.map_cons, List.sum_cons,
          termsForm_eval_any assignment rest]
        simp [identityEnv, inside]

theorem identityEnv_sourceEnv {width logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth) :
    identityEnv (fun column : Fin width => LogicalPlan.sourceEnv layout assignment column.val) =
      LogicalPlan.sourceEnv layout assignment := by
  funext column
  by_cases inside : column < width
  · simp [identityEnv, inside]
  · simp [identityEnv, LogicalPlan.sourceEnv, inside]

/-- `CircuitRelation.formOf_ofVar_eval` on a source environment, for every column. -/
theorem formOf_ofVar_sourceEnv {width logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (column : Nat) :
    (formOf (width := width) (R1CS.LinearCombination.ofVar column)).eval
        (fun source : Fin width => LogicalPlan.sourceEnv layout assignment source.val) =
      LogicalPlan.sourceEnv layout assignment column := by
  rw [formOf, termsForm_eval_any, identityEnv_sourceEnv]
  simp [R1CS.LinearCombination.ofVar]

/-- The S-box rows and the binding rows give each block's outputs as the permutation of its
inputs: `CircuitRelation.outputsHold` for any blocks and S-box start, without the width premise. -/
theorem outputsHold_of_rows {width logicalWidth : Nat} (blocks : List Block) (sboxStart : Nat)
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment layout.one = 1) (rowCount_le : blocks.length * 150 ≤ 2 ^ cubeVariables)
    (sboxRows : (PoseidonSboxFamilyPlan.plan (sboxInterface layout blocks sboxStart)
      rowCount_le).RowsZero assignment)
    (binding : R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment)
      (bindingRows width blocks sboxStart)) :
    OutputsHold blocks (LogicalPlan.sourceEnv layout assignment) := by
  intro block member
  obtain ⟨index, rfl⟩ := List.mem_iff_get.mp member
  have permuted := PoseidonSboxFamilyPlan.planRowsZero_implies_permute _ _ assignment one
    sboxRows index
  have inputs : SparseLayer.evalState assignment
      ((sboxInterface layout blocks sboxStart).input index) =
      (blocks.get index).input (LogicalPlan.sourceEnv layout assignment) := by
    funext lane
    exact layout.combination_eval assignment one _
  have finals : SparseLayer.evalState assignment
      (fun lane => (sboxInterface layout blocks sboxStart).sboxOutput index
        (PoseidonRetainedSlots.finalRow lane)) =
      SparseLayer.evalState (fun column : Fin width =>
          LogicalPlan.sourceEnv layout assignment column.val)
        (fun lane => sboxOutputs (width := width) blocks sboxStart index
          (PoseidonRetainedSlots.finalRow lane)) := by
    funext lane
    simp only [SparseLayer.evalState, sboxInterface, sboxOutputs, formOf_ofVar_sourceEnv,
      LogicalPlan.Layout.form_eval]
  have outputs : SparseLayer.evalState assignment
      ((sboxInterface layout blocks sboxStart).output index) =
      (blocks.get index).output (LogicalPlan.sourceEnv layout assignment) := by
    funext lane
    have row := binding _ (List.mem_flatMap.mpr ⟨index, List.mem_finRange index,
      List.mem_ofFn.mpr ⟨lane, rfl⟩⟩)
    simp only [R1CS.Row.Holds, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
      R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_one,
      R1CS.LinearCombination.eval_zero, mul_one, toCombination_eval] at row
    rw [neg_one_mul, ← sub_eq_add_neg, sub_eq_zero] at row
    simp only [Block.output, SparseLayer.evalState, PoseidonSboxFamilyPlan.Interface.output,
      SparseLayer.eval_external]
    have external := congrArg (fun state => Layer.externalF state lane) finals
    rw [external, row]
    simp only [outputState, SparseLayer.eval_external]
  rw [inputs, outputs] at permuted
  apply List.ofFn_injective
  rw [permuted, Permutation.runF_eq_reference, Permutation.runReference_schedule]

/-- A block's outputs as `Spec.Poseidon2.permute` of its lanes. -/
theorem output_permute {blocks : List Block} {env : Env} (outputs : OutputsHold blocks env)
    {block : Block} (member : block ∈ blocks) (linear : LinearLanes block.lanes) :
    List.ofFn (block.output env) =
      Spec.Poseidon2.permute
        (List.ofFn fun lane : Fin 16 => (block.lanes.getD lane.val 0).eval env) := by
  have inputs : (fun lane : Fin 16 => (block.lanes.getD lane.val 0).eval env) =
      block.input env := by
    funext lane
    exact lane_eval block.lanes linear env lane.val
  rw [inputs, outputs block member, Permutation.runF_eq_reference,
    Permutation.runReference_schedule]

/-- Rows that read no dead local, with each block's outputs the permutation of its inputs, hold in
an environment that differs from `env` only on the dead locals, where each block's 1,096 template
rows and template constraints also hold. -/
theorem completed_rows {blocks : List Block} (frame : Framed blocks) {rows : List R1CS.Row}
    (avoid : rows.all (rowAvoids (deadIn blocks)) = true) {env : Env}
    (rowsHold : R1CS.RowsHold env rows) (outputs : OutputsHold blocks env) :
    ∃ completed : Env,
      (∀ column, deadIn blocks column = false → completed column = env column) ∧
      R1CS.RowsHold completed rows ∧
      ∀ block ∈ blocks,
        PermutationInvocationHolds (PilotData.circuitPackage ()) block.invocation completed ∧
        ConstraintsHold completed ((Piece.permutation block.start block.lanes).constraints
          (PilotData.canonicalConstraints ())) := by
  have completed := complete_sound frame env outputs
  refine ⟨complete blocks env, completed.1, rowsHold_congr completed.1 avoid rowsHold, ?_⟩
  intro block member
  exact ⟨completed.2 block member, permutation_sound 0 0 block.start block.lanes
    (frame.linear block member) _ (completed.2 block member)⟩

/-- Soundness of native blocks. The rows are a package's rows before the binding rows
(`StepPackage.Package.relationRows` is `rows ++ bindingRows`). -/
theorem blocks_sound {width logicalWidth : Nat} {blocks : List Block} (frame : Framed blocks)
    (sboxStart : Nat) {rows : List R1CS.Row} (avoid : rows.all (rowAvoids (deadIn blocks)) = true)
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment layout.one = 1) (rowCount_le : blocks.length * 150 ≤ 2 ^ cubeVariables)
    (sboxRows : (PoseidonSboxFamilyPlan.plan (sboxInterface layout blocks sboxStart)
      rowCount_le).RowsZero assignment)
    (rowsHold : R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment)
      (rows ++ bindingRows width blocks sboxStart)) :
    (∀ block ∈ blocks, List.ofFn (block.output (LogicalPlan.sourceEnv layout assignment)) =
      Spec.Poseidon2.permute (List.ofFn fun lane : Fin 16 =>
        (block.lanes.getD lane.val 0).eval (LogicalPlan.sourceEnv layout assignment))) ∧
    ∃ completed : Env,
      (∀ column, deadIn blocks column = false →
        completed column = LogicalPlan.sourceEnv layout assignment column) ∧
      R1CS.RowsHold completed rows ∧
      ∀ block ∈ blocks,
        PermutationInvocationHolds (PilotData.circuitPackage ()) block.invocation completed ∧
        ConstraintsHold completed ((Piece.permutation block.start block.lanes).constraints
          (PilotData.canonicalConstraints ())) := by
  rw [R1CS.rowsHold_append] at rowsHold
  have outputs := outputsHold_of_rows blocks sboxStart layout assignment one rowCount_le sboxRows
    rowsHold.2
  exact ⟨fun block member => output_permute outputs member (frame.linear block member),
    completed_rows frame avoid rowsHold.1 outputs⟩

/-! ### Completeness -/

/-- The local of `block` that holds S-box output `row` of the template. -/
def sboxLocal (block : Block) (row : Fin PoseidonRetainedSlots.rows.length) : Nat :=
  block.start + (PoseidonRetainedSlots.localOutput row).val

/-- `PoseidonSboxSourceCompleteness.equations_of_sourceRows` for a substituted block: when the
block's template constraints hold in `honest`, the inputs read its lanes there and each S-box
output reads its template local, the S-box equations hold. -/
theorem sboxEquations_of_block {logicalWidth : Nat}
    (interface : PoseidonSboxPlan.Interface logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment interface.oneColumn = 1) (block : Block) (honest : Env)
    (inputs : ∀ lane : Fin 16,
      (interface.input lane).eval assignment = (block.lanes.getD lane.val 0).eval honest)
    (sboxes : ∀ row, (interface.sboxOutput row).eval assignment = honest (sboxLocal block row))
    (templates : ConstraintsHold honest ((Piece.permutation block.start block.lanes).constraints
      (PilotData.canonicalConstraints ()))) :
    PoseidonSboxPlan.SboxEquations interface assignment := by
  have substituted : ConstraintsHold honest ((PilotData.canonicalConstraints ()).map
      (subst (laneSubst 16 block.start block.lanes))) := templates
  have canonical : ConstraintsHold
      (fun column => (laneSubst 16 block.start block.lanes column).eval honest)
      (PilotData.canonicalConstraints ()) := by
    intro pattern member
    have value := substituted _ (List.mem_map_of_mem member)
    rwa [Lifecycle.Nebula.Expr.eval_subst] at value
  apply PoseidonSboxSourceCompleteness.equations_of_sourceRows interface assignment
    (fun column => (laneSubst 16 block.start block.lanes column).eval honest) one _ _ canonical
  · funext lane
    show (interface.input lane).eval assignment =
      (laneSubst 16 block.start block.lanes lane.val).eval honest
    rw [inputs lane, laneSubst_input block.lanes lane.isLt]
  · intro row
    rw [sboxes row, PoseidonRetainedSlots.output_eq_input_add_local]
    show honest (sboxLocal block row) = (laneSubst 16 block.start block.lanes
      (16 + (PoseidonRetainedSlots.localOutput row).val)).eval honest
    rw [laneSubst_local]
    rfl

theorem lane_avoids {blocks : List Block} (frame : Framed blocks) {block : Block}
    (member : block ∈ blocks) (lane : Nat) :
    avoids (fun column => block.covers column || deadIn blocks column)
      (block.lanes.getD lane 0) = true := by
  by_cases inside : lane < block.lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact frame.lanes block member _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

/-- A block's output locals are no block's dead locals. -/
theorem output_live {blocks : List Block} (frame : Framed blocks) {block : Block}
    (member : block ∈ blocks) (lane : Fin 16) :
    deadIn blocks (block.start + 1080 + lane.val) = false := by
  have below := lane.isLt
  simp only [deadIn, List.any_eq_false, Block.dead, Bool.and_eq_true, decide_eq_true_eq]
  intro other otherMember inside
  by_cases same : other = block
  · subst same
    omega
  · have separated := separated_of_mem frame.separated otherMember member same
    unfold Separated at separated
    omega

/-- Completeness of native blocks: an honest environment where every block's template
constraints hold, read by the assignment off the dead locals, with each S-box column at its
template local, makes the S-box rows zero and the binding rows hold. -/
theorem blocks_complete {width logicalWidth : Nat} {blocks : List Block} (frame : Framed blocks)
    (sboxStart : Nat) (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment layout.one = 1)
    (rowCount_le : blocks.length * 150 ≤ 2 ^ cubeVariables) (honest : Env)
    (templates : ∀ block ∈ blocks, ConstraintsHold honest
      ((Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())))
    (live : ∀ column, deadIn blocks column = false →
      LogicalPlan.sourceEnv layout assignment column = honest column)
    (sboxes : ∀ (index : Fin blocks.length) (row : Fin PoseidonRetainedSlots.rows.length),
      LogicalPlan.sourceEnv layout assignment (sboxColumn sboxStart index.val row.val) =
        honest (sboxLocal (blocks.get index) row)) :
    (PoseidonSboxFamilyPlan.plan (sboxInterface layout blocks sboxStart) rowCount_le).RowsZero
        assignment ∧
      R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment)
        (bindingRows width blocks sboxStart) := by
  constructor
  · apply PoseidonSboxFamilyPlan.equations_imply_planRowsZero
    intro index
    have member := List.get_mem blocks index
    refine ⟨?_, fun lane => by rw [PoseidonSboxFamilyPlan.invocation_output_eq_trace]⟩
    apply sboxEquations_of_block _ assignment one (blocks.get index) honest _ _
      (templates _ member)
    · intro lane
      show (layout.combination
        (invocationInputCombination (blocks.get index).invocation lane.val).toR1CS).eval
          assignment = _
      rw [layout.combination_eval assignment one]
      change (blocks.get index).input (LogicalPlan.sourceEnv layout assignment) lane = _
      rw [input_eq frame member]
      apply eval_congr (dead := fun column => (blocks.get index).covers column || deadIn blocks column)
      · intro column avoid
        simp only [Bool.or_eq_false_iff] at avoid
        exact live column avoid.2
      · exact lane_avoids frame member lane.val
    · intro row
      show (layout.form (sboxColumn sboxStart index.val row.val)).eval assignment = _
      rw [LogicalPlan.Layout.form_eval]
      exact sboxes index row
  · intro row member
    obtain ⟨index, _, rowMember⟩ := List.mem_flatMap.mp member
    obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp rowMember
    have blockMember := List.get_mem blocks index
    have invocationHolds := (constraints_iff_invocation 0 0 _ _ (frame.linear _ blockMember)
      honest).mp (templates _ blockMember)
    have final : honest ((blocks.get index).start + 1080 + lane.val) =
        Layer.externalF (fun lane => honest
          (sboxLocal (blocks.get index) (PoseidonRetainedSlots.finalRow lane))) lane :=
      congrFun (PermutationOutput.invocation_finalLayer _ honest invocationHolds) lane
    have finals : SparseLayer.evalState
        (fun column : Fin width => LogicalPlan.sourceEnv layout assignment column.val)
        (fun lane => sboxOutputs (width := width) blocks sboxStart index
          (PoseidonRetainedSlots.finalRow lane)) =
        fun lane => honest (sboxLocal (blocks.get index) (PoseidonRetainedSlots.finalRow lane)) := by
      funext lane
      simp only [SparseLayer.evalState, sboxOutputs, formOf_ofVar_sourceEnv]
      exact sboxes index _
    simp only [R1CS.Row.Holds, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
      R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_one,
      R1CS.LinearCombination.eval_zero, mul_one, toCombination_eval]
    rw [neg_one_mul, ← sub_eq_add_neg, sub_eq_zero, live _ (output_live frame blockMember lane)]
    simp only [outputState, SparseLayer.eval_external]
    rw [finals]
    exact final

/-! ### Bit slots -/

/-- The honest value of a column that a `bit` hint fills is 0 or 1. -/
theorem bitHint_valid (env : Env) (source : Expr) (index : Nat) :
    LowNormSlot.Valid .bit (Hint.eval env (.bit source index)) := by
  show Hint.ofNat (((source.eval env).val >>> index) &&& 1) = 0 ∨
    Hint.ofNat (((source.eval env).val >>> index) &&& 1) = 1
  rw [Nat.and_one_is_mod]
  rcases Nat.mod_two_eq_zero_or_one ((source.eval env).val >>> index) with bit | bit <;>
    rw [bit]
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- A value that satisfies the row `b * b = b` is 0 or 1. -/
theorem bit_valid_of_boolean {value : F} (boolean : value * value = value) :
    LowNormSlot.Valid .bit value := by
  have product : value * (value - 1) = 0 := by
    rw [mul_sub, mul_one, boolean, sub_self]
  rcases GoldilocksPrime.baseFieldNoZeroDivisors value (value - 1) product with zero | minusOne
  · exact Or.inl zero
  · exact Or.inr (sub_eq_zero.mp minusOne)

private theorem ofFn_eq_singleton {α : Type} {count : Nat} (values : Fin count → α)
    (one : count = 1) : List.ofFn values = [values ⟨0, by omega⟩] := by
  subst one
  rfl

theorem bitSlot_width {sourceWidth : Nat} {slot : LowNormAssignment.Slot sourceWidth}
    (bit : slot.kind = .bit) : slot.width = 1 := by
  simp [LowNormAssignment.Slot.width, bit, LowNormSlot.Kind.width]

/-- A `bit` slot decodes to its one coordinate, in any logical assignment. -/
theorem bitSlot_form_eval {sourceWidth : Nat} (slots : List (LowNormAssignment.Slot sourceWidth))
    (slot : Fin slots.length) (bit : (slots.get slot).kind = .bit)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots)) :
    (RetainedSlot.form slots slot).eval assignment =
      assignment (ProductionAssignment.privateColumn
        (LowNormAssignment.coordinateIndex slots slot ⟨0, by rw [bitSlot_width bit]; omega⟩)) := by
  rw [RetainedSlot.form, RetainedSlot.coordinateForms, ofFn_eq_singleton _ (bitSlot_width bit),
    RetainedSlot.recomposeForms_eval]
  simp only [List.map_cons, List.map_nil, SparseForm.singleton_eval, one_mul]
  exact LowNormSlot.recompose_encode .bit _

theorem representedMinusOne_eq :
    Spec.Folding.PiCCS.PaperJoint.NormRange.representedMinusOne = -1 := by
  decide

/-- A `bit` slot whose coordinate passes the norm check decodes to -1, 0 or 1. -/
theorem bitSlot_sound {sourceWidth : Nat} (slots : List (LowNormAssignment.Slot sourceWidth))
    (slot : Fin slots.length) (bit : (slots.get slot).kind = .bit)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots))
    (norm : ∀ column, centeredMagnitude (assignment column) < 2) :
    (RetainedSlot.form slots slot).eval assignment = -1 ∨
      (RetainedSlot.form slots slot).eval assignment = 0 ∨
      (RetainedSlot.form slots slot).eval assignment = 1 := by
  rw [bitSlot_form_eval slots slot bit]
  rcases (Spec.Folding.PiCCS.PaperJoint.NormRange.strictNormTwo_iff_representedRoot _).mp
      (norm _) with minusOne | zero | one
  · exact Or.inl (minusOne.trans representedMinusOne_eq)
  · exact Or.inr (Or.inl zero)
  · exact Or.inr (Or.inr one)

end NightstreamFPrime.Export.Generic.NativeBlock
