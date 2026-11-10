import NightstreamFPrime.Export.Generic.Frame
import NightstreamFPrime.Export.Generic.LogicalPlan
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan

/-! The logical relation of a step circuit (CW-5.8). The circuit's constraints, with the decode
constraints of its output digest, are split into pieces (`Pieces.split`). When the Poseidon2
blocks pack (`packable`: the frame check, every block after the public columns, and no other piece
reading a dead local), the relation is the R1CS rows of the other pieces (`PieceRows.lowerPieces`),
16 rows per block that bind its output locals, and today's `PoseidonSboxFamilyPlan`: 150 S-box rows
per block on fresh columns. Otherwise every piece is lowered to R1CS rows. Soundness never depends
on the check: `rowsZero_implies_spec` gives the circuit's spec and the output digest in an
environment that differs from the assignment only on dead locals (`Frame.complete`). -/

namespace NightstreamFPrime.Export.Generic.CircuitRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2

/-! ### Blocks and the other pieces -/

def isBlock : Piece → Bool
  | .permutation .. => true
  | _ => false

def blocksOf : List Piece → List Block
  | [] => []
  | .permutation start lanes :: rest => ⟨start, lanes⟩ :: blocksOf rest
  | _ :: rest => blocksOf rest

def otherPieces (pieces : List Piece) : List Piece :=
  pieces.filter fun piece => !isBlock piece

theorem mem_blocksOf {start : Nat} {lanes : List Expr} :
    ∀ {pieces : List Piece}, Piece.permutation start lanes ∈ pieces →
      (⟨start, lanes⟩ : Block) ∈ blocksOf pieces
  | [], member => by simp at member
  | piece :: rest, member => by
      rcases List.mem_cons.mp member with equal | restMember
      · subst equal
        exact List.mem_cons_self
      · have inRest := mem_blocksOf restMember
        cases piece with
        | permutation _ _ => exact List.mem_cons_of_mem _ inRest
        | constraint _ => exact inRest
        | ringProduct _ _ _ => exact inRest

/-- A piece outside the blocks reads no column that `dead` marks. -/
def avoidsDead (dead : Nat → Bool) : Piece → Bool
  | .constraint expression => avoids dead expression
  | .permutation .. => true
  | .ringProduct cells start lanes =>
      lanes.all (avoids dead) &&
        (List.range (ringDegree * cells)).all fun index => !dead (start + index)

/-- Every block starts at or after `bound`. -/
def startsFrom (bound : Nat) (pieces : List Piece) : Bool :=
  (blocksOf pieces).all fun block => decide (bound ≤ block.start)

/-- No column below the blocks' start is a dead local. -/
theorem live_below {bound : Nat} {pieces : List Piece} (starts : startsFrom bound pieces = true)
    {column : Nat} (below : column < bound) : deadIn (blocksOf pieces) column = false := by
  simp only [startsFrom, List.all_eq_true, decide_eq_true_eq] at starts
  simp only [deadIn, List.any_eq_false, Block.dead, Bool.and_eq_true, decide_eq_true_eq]
  intro block member dead
  have := starts block member
  omega

/-- The blocks pack: they pass the frame check, start after the public columns, and no other
piece reads a dead local. -/
def packable (publicColumns : Nat) (pieces : List Piece) : Bool :=
  checkFrame (blocksOf pieces) && startsFrom publicColumns pieces &&
    pieces.all (avoidsDead (deadIn (blocksOf pieces)))

theorem varsBelow_sub {left right : Expr} {bound : Nat} (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) : (left - right).VarsBelow bound :=
  ⟨leftBelow, trivial, rightBelow⟩

theorem varsBelow_add {left right : Expr} {bound : Nat} (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) : (left + right).VarsBelow bound :=
  ⟨leftBelow, rightBelow⟩

theorem ringConstraint_varsBelow (cells : Nat) (index : Fin (ringDegree * cells)) :
    (ringConstraint cells index).VarsBelow (ringInputs cells + ringDegree * cells) := by
  have positive : 0 < cells := by
    rcases Nat.eq_zero_or_pos cells with zero | positive
    · have below := index.isLt
      simp [zero] at below
    · exact positive
  have cellBelow := Nat.mod_lt index.val positive
  have indexBelow := index.isLt
  have slot := PieceRows.slot_lt (ringLane index) cellBelow
  unfold ringConstraint
  refine varsBelow_sub ?_ (varsBelow_add ?_ ?_)
  · show ringInputs cells + index.val < ringInputs cells + ringDegree * cells
    omega
  · show ringPrior cells (ringLane index) (index.val % cells) <
      ringInputs cells + ringDegree * cells
    simp only [ringPrior, ringInputs]
    omega
  · apply Lifecycle.PiRLC.v1_2.CombinationStep.mulExpr_varsBelow
    · intro source
      show source.val < ringInputs cells + ringDegree * cells
      have := source.isLt
      simp only [ringInputs]
      omega
    · intro source
      show ringValue cells source (index.val % cells) < ringInputs cells + ringDegree * cells
      have := PieceRows.slot_lt source cellBelow
      simp only [ringValue, ringInputs]
      omega

/-- A piece outside the blocks holds wherever it held, after a change of dead columns only. -/
theorem avoidsDead_congr {dead : Nat → Bool} {before after : Env}
    (agree : ∀ column, dead column = false → after column = before column)
    (canonical : List Expr) :
    ∀ piece : Piece, isBlock piece = false → avoidsDead dead piece = true →
      ConstraintsHold before (piece.constraints canonical) →
        ConstraintsHold after (piece.constraints canonical)
  | .constraint expression, _, avoid, holds => by
      intro current member
      simp only [Piece.constraints, List.mem_singleton] at member
      subst member
      rw [eval_congr agree current avoid]
      exact holds current List.mem_cons_self
  | .permutation .., notBlock, _, _ => by simp [isBlock] at notBlock
  | .ringProduct cells start lanes, _, avoid, holds => by
      simp only [avoidsDead, Bool.and_eq_true, List.all_eq_true, List.mem_range] at avoid
      intro current member
      have original := holds current member
      simp only [Piece.constraints, ringTemplate, List.mem_map, List.mem_ofFn] at member
      obtain ⟨_, ⟨index, rfl⟩, rfl⟩ := member
      have wiresAvoid : ∀ column, column < ringInputs cells + ringDegree * cells →
          avoids dead (laneSubst (ringInputs cells) start lanes column) = true := by
        intro column below
        by_cases input : column < ringInputs cells
        · rw [laneSubst_input lanes input]
          by_cases inside : column < lanes.length
          · rw [List.getD_eq_getElem _ _ inside]
            exact avoid.1 _ (List.getElem_mem inside)
          · rw [List.getD_eq_default _ _ (by omega)]
            rfl
        · have live := avoid.2 (column - ringInputs cells) (by omega)
          simp only [laneSubst, input, ite_false, avoids]
          simpa using live
      rw [eval_congr agree _ (avoids_subst _ wiresAvoid _ (ringConstraint_varsBelow cells index))]
      exact original

/-! ### Soundness on pieces -/

theorem packable_framed {publicColumns : Nat} {pieces : List Piece}
    (check : packable publicColumns pieces = true) : Framed (blocksOf pieces) := by
  simp only [packable, Bool.and_eq_true] at check
  exact checkFrame_spec check.1.1

theorem packable_public {publicColumns : Nat} {pieces : List Piece}
    (check : packable publicColumns pieces = true) {column : Nat} (below : column < publicColumns) :
    deadIn (blocksOf pieces) column = false := by
  simp only [packable, Bool.and_eq_true] at check
  exact live_below check.1.2 below

/-- When the blocks pack, every piece holds in the completed environment: the blocks by their
honest traces, the other pieces because they read no dead local. -/
theorem packed_holds {publicColumns : Nat} {pieces : List Piece}
    (valid : ∀ piece ∈ pieces, piece.Valid) (check : packable publicColumns pieces = true)
    (env : Env)
    (others : ∀ piece ∈ pieces, isBlock piece = false →
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (outputs : OutputsHold (blocksOf pieces) env) :
    ∀ piece ∈ pieces, ConstraintsHold (complete (blocksOf pieces) env)
      (piece.constraints (PilotData.canonicalConstraints ())) := by
  have completed := complete_sound (packable_framed check) env outputs
  simp only [packable, Bool.and_eq_true, List.all_eq_true] at check
  intro piece member
  cases piece with
  | permutation start lanes =>
      have blockHolds : PermutationInvocationHolds (PilotData.circuitPackage ())
          (invocation 0 0 start lanes) (complete (blocksOf pieces) env) :=
        completed.2 ⟨start, lanes⟩ (mem_blocksOf member)
      exact permutation_sound 0 0 start lanes (valid _ member) (complete (blocksOf pieces) env)
        blockHolds
  | constraint expression =>
      exact avoidsDead_congr completed.1 _ _ rfl (check.2 _ member) (others _ member rfl)
  | ringProduct cells start lanes =>
      exact avoidsDead_congr completed.1 _ _ rfl (check.2 _ member) (others _ member rfl)

/-- The checks that replace the S-box and binding rows: the rows of the pieces other than the
blocks, and each block's outputs as the permutation of its inputs. When the blocks pack, every
piece holds in the completed environment, which differs from `env` only on dead locals. -/
theorem checks_implies_pieces {publicColumns : Nat} {pieces : List Piece}
    (valid : ∀ piece ∈ pieces, piece.Valid) (packed : packable publicColumns pieces = true)
    (next : Nat) (env : Env)
    (rows : R1CS.RowsHold env
      (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1)
    (outputs : OutputsHold (blocksOf pieces) env) :
    (∀ piece ∈ pieces, ConstraintsHold (complete (blocksOf pieces) env)
        (piece.constraints (PilotData.canonicalConstraints ()))) ∧
      ∀ column, deadIn (blocksOf pieces) column = false →
        complete (blocksOf pieces) env column = env column := by
  have others := (PieceRows.lowerPiecesAux_sound _ env (otherPieces pieces) next []
    (fun piece member => valid piece (List.mem_filter.mp member).1) rows).2
  exact ⟨packed_holds valid packed env (fun piece member notBlock =>
      others piece (List.mem_filter.mpr ⟨member, by simp [notBlock]⟩)) outputs,
    fun _ live => complete_live (packable_framed packed) outputs live⟩

/-- The pieces that `split` cut give the constraints. -/
theorem split_holds {constraints : List Expr} {env : Env}
    (holds : ∀ piece ∈ split (PilotData.canonicalConstraints ()) constraints,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    ConstraintsHold env constraints := by
  intro expression member
  rw [← flatMap_split (PilotData.canonicalConstraints ()) constraints, List.mem_flatMap] at member
  obtain ⟨piece, pieceMember, inPiece⟩ := member
  exact holds piece pieceMember expression inPiece

/-! ### Forms on the logical columns -/

/-- Column `c` of an assignment as an environment, 0 beyond the width. -/
def identityEnv {width : Nat} (assignment : Assignment F width) : Env :=
  fun column => if bounded : column < width then assignment ⟨column, bounded⟩ else 0

/-- Sparse terms as a form on the logical columns; terms outside the width are dropped. -/
def termsForm {width : Nat} : List (Nat × F) → SparseForm width
  | [] => .empty
  | (column, coefficient) :: rest =>
      if bounded : column < width then
        .add (.singleton ⟨column, bounded⟩ coefficient) (termsForm rest)
      else termsForm rest

theorem termsForm_eval {width : Nat} (assignment : Assignment F width) :
    ∀ terms : List (Nat × F), (∀ term ∈ terms, term.1 < width) →
      (termsForm terms).eval assignment =
        (terms.map fun term => term.2 * identityEnv assignment term.1).sum
  | [], _ => by simp [termsForm]
  | (column, coefficient) :: rest, bounded => by
      have inside : column < width := bounded _ List.mem_cons_self
      have restEval := termsForm_eval assignment rest
        (fun term member => bounded term (List.mem_cons_of_mem _ member))
      simp only [termsForm, inside, dite_true, SparseForm.add_eval, SparseForm.singleton_eval,
        List.map_cons, List.sum_cons, restEval]
      simp [identityEnv, inside]

/-- A linear combination as a form; its constant reads the constant column 0. -/
def formOf {width : Nat} (combination : R1CS.LinearCombination) : SparseForm width :=
  termsForm ((0, combination.constant) :: combination.terms)

/-- `formOf` of one column reads that column. -/
theorem formOf_ofVar_eval {width : Nat} (assignment : Assignment F width) (column : Nat)
    (bounded : column < width) :
    (formOf (width := width) (R1CS.LinearCombination.ofVar column)).eval assignment =
      assignment ⟨column, bounded⟩ := by
  have positive : 0 < width := by omega
  rw [formOf, termsForm_eval assignment _ (by
    intro term member
    rcases List.mem_cons.mp member with rfl | member
    · exact positive
    · simp [R1CS.LinearCombination.ofVar] at member
      rw [member]
      exact bounded)]
  simp [R1CS.LinearCombination.ofVar, identityEnv, bounded]

/-- A form as an R1CS combination over the same columns. -/
def toCombination {width : Nat} (form : SparseForm width) : R1CS.LinearCombination :=
  ⟨0, form.entries.map fun entry => (entry.column.val, entry.coefficient)⟩

theorem toCombination_eval {width : Nat} (form : SparseForm width) (env : Env) :
    (toCombination form).eval env = form.eval (fun column => env column.val) := by
  rcases form with ⟨entries⟩
  induction entries with
  | nil =>
      have empty : (⟨[]⟩ : SparseForm width).eval (fun column => env column.val) = 0 :=
        SparseForm.empty_eval _
      rw [empty]
      simp [toCombination, R1CS.LinearCombination.eval]
  | cons entry rest inductionHypothesis =>
      have split : (⟨entry :: rest⟩ : SparseForm width).eval (fun column => env column.val) =
          entry.coefficient * env entry.column.val +
            (⟨rest⟩ : SparseForm width).eval (fun column => env column.val) := by
        rw [show (⟨entry :: rest⟩ : SparseForm width) =
            .add (.singleton entry.column entry.coefficient) ⟨rest⟩ from rfl,
          SparseForm.add_eval, SparseForm.singleton_eval]
      rw [split, ← inductionHypothesis]
      simp only [toCombination, R1CS.LinearCombination.eval, List.map_cons, List.map_map,
        List.sum_cons, Function.comp_def, zero_add]

/-! ### The packed relation -/

/-- Fresh column of S-box output `row` of block `block`, from `sboxStart`. -/
def sboxColumn (sboxStart block row : Nat) : Nat :=
  sboxStart + (block * 150 + row)

def sboxOutputs {width : Nat} (blocks : List Block) (sboxStart : Nat) :
    Fin blocks.length → Fin PoseidonRetainedSlots.rows.length → SparseForm width :=
  fun index row => formOf (R1CS.LinearCombination.ofVar (sboxColumn sboxStart index.val row.val))

/-- The S-box family's interface: each block's input lanes and S-box outputs read through the
layout. -/
def sboxInterface {width logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (blocks : List Block) (sboxStart : Nat) :
    PoseidonSboxFamilyPlan.Interface logicalWidth blocks.length where
  oneColumn := layout.one
  input := fun index lane =>
    layout.combination (invocationInputCombination (blocks.get index).invocation lane.val).toR1CS
  sboxOutput := fun index row => layout.form (sboxColumn sboxStart index.val row.val)

/-- The output state of a block, from its final S-box outputs (`Interface.output`). -/
def outputState {width : Nat} (blocks : List Block) (sboxStart : Nat)
    (index : Fin blocks.length) : PoseidonSboxPlan.State width :=
  SparseLayer.external fun lane =>
    sboxOutputs blocks sboxStart index (PoseidonRetainedSlots.finalRow lane)

/-- Row `output local - output form = 0` for each output lane of each block. -/
def bindingRows (width : Nat) (blocks : List Block) (sboxStart : Nat) : List R1CS.Row :=
  (List.finRange blocks.length).flatMap fun index => List.ofFn fun lane : Fin 16 =>
    ⟨.add (.ofVar ((blocks.get index).start + 1080 + lane.val))
        (.scale (-1) (toCombination (outputState (width := width) blocks sboxStart index lane))),
      .one, .zero⟩

/-- The first fresh S-box column: after the fresh columns of the other pieces. -/
def sboxStart (pieces : List Piece) (next : Nat) : Nat :=
  (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).2

/-- The next fresh column after the pieces, with no ring row built. -/
def nextAfter (canonical : List Expr) : List Piece → Nat → Nat
  | [], next => next
  | .ringProduct cells _ _ :: rest, next => nextAfter canonical rest (next + ringDegree * cells)
  | piece :: rest, next =>
      nextAfter canonical rest (R1CS.lowerConstraints (piece.constraints canonical) next).next

theorem lowerPiecesAux_next (canonical : List Expr) :
    ∀ (pieces : List Piece) (next : Nat) (rowsReversed : List R1CS.Row),
      (PieceRows.lowerPiecesAux canonical pieces next rowsReversed).2 =
        nextAfter canonical pieces next
  | [], _, _ => rfl
  | piece :: rest, next, rowsReversed => by
      cases piece <;>
        simp only [PieceRows.lowerPiecesAux, PieceRows.lowerPiece, nextAfter,
          lowerPiecesAux_next canonical rest]

/-- Compiled code finds the first fresh S-box column and builds no ring row. -/
def sboxStartFast (pieces : List Piece) (next : Nat) : Nat :=
  nextAfter (PilotData.canonicalConstraints ()) (otherPieces pieces) next

@[csimp] theorem sboxStart_eq_sboxStartFast : @sboxStart = @sboxStartFast := by
  funext pieces next
  rw [sboxStart, PieceRows.lowerPieces, lowerPiecesAux_next, sboxStartFast]

/-- The R1CS rows of the relation: the other pieces and the binding rows when the blocks pack,
every piece otherwise. -/
def relationRows (publicColumns : Nat) (pieces : List Piece) (next width : Nat) :
    List R1CS.Row :=
  if packable publicColumns pieces then
    (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1 ++
      bindingRows width (blocksOf pieces) (sboxStart pieces next)
  else (PieceRows.lowerPieces (PilotData.canonicalConstraints ()) pieces next).1

structure BlockFits (pieces : List Piece) (next width rowCount : Nat) : Prop where
  inputs : ∀ block ∈ blocksOf pieces, ∀ lane : Fin 16,
    ∀ term ∈ (invocationInputCombination block.invocation lane.val).toR1CS.terms, term.1 < width
  sboxColumns : sboxStart pieces next + (blocksOf pieces).length * 150 ≤ width
  rowCount : rowCount + (blocksOf pieces).length * 150 ≤ 2 ^ cubeVariables

/-- The relation fits the cube and the logical width. -/
structure Fits (publicColumns : Nat) (pieces : List Piece) (next width : Nat) : Prop where
  rows : LogicalPlan.Fits (relationRows publicColumns pieces next width) width
  blocks : packable publicColumns pieces = true →
    BlockFits pieces next width (relationRows publicColumns pieces next width).length

theorem ordinary_rowCount (rows : List R1CS.Row) (width : Nat) (fits : LogicalPlan.Fits rows width)
    {logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth) :
    (LogicalPlan.plan rows width fits layout).rowCount = rows.length :=
  rfl

def plan (publicColumns : Nat) (pieces : List Piece) (next width : Nat)
    (fits : Fits publicColumns pieces next width) {logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) : Plan logicalWidth :=
  if packed : packable publicColumns pieces = true then
    Plan.append (LogicalPlan.plan _ width fits.rows layout)
      (PoseidonSboxFamilyPlan.plan
        (sboxInterface layout (blocksOf pieces) (sboxStart pieces next))
        (by have := (fits.blocks packed).rowCount; omega))
      (by
        rw [ordinary_rowCount, PoseidonSboxFamilyPlan.plan_rowCount]
        exact (fits.blocks packed).rowCount)
  else LogicalPlan.plan _ width fits.rows layout

/-- Binding rows and S-box rows give each block's outputs as the permutation of its inputs. -/
theorem outputsHold {publicColumns : Nat} {pieces : List Piece} {next width logicalWidth : Nat}
    (fits : Fits publicColumns pieces next width) (packed : packable publicColumns pieces = true)
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment layout.one = 1)
    (binding : R1CS.RowsHold (LogicalPlan.sourceEnv layout assignment)
      (bindingRows width (blocksOf pieces) (sboxStart pieces next)))
    (sboxRows : (PoseidonSboxFamilyPlan.plan
        (sboxInterface layout (blocksOf pieces) (sboxStart pieces next))
        (by have := (fits.blocks packed).rowCount; omega)).RowsZero assignment) :
    OutputsHold (blocksOf pieces) (LogicalPlan.sourceEnv layout assignment) := by
  intro block member
  obtain ⟨index, rfl⟩ := List.mem_iff_get.mp member
  have permuted := PoseidonSboxFamilyPlan.planRowsZero_implies_permute _ _ assignment one sboxRows
    index
  have inputs : SparseLayer.evalState assignment
      ((sboxInterface layout (blocksOf pieces) (sboxStart pieces next)).input index) =
      ((blocksOf pieces).get index).input (LogicalPlan.sourceEnv layout assignment) := by
    funext lane
    exact layout.combination_eval assignment one _
  have finals : SparseLayer.evalState assignment
      (fun lane => (sboxInterface layout (blocksOf pieces) (sboxStart pieces next)).sboxOutput
        index (PoseidonRetainedSlots.finalRow lane)) =
      SparseLayer.evalState (fun column : Fin width =>
          LogicalPlan.sourceEnv layout assignment column.val)
        (fun lane => sboxOutputs (width := width) (blocksOf pieces) (sboxStart pieces next) index
          (PoseidonRetainedSlots.finalRow lane)) := by
    funext lane
    have bound : sboxColumn (sboxStart pieces next) index.val
        (PoseidonRetainedSlots.finalRow lane).val < width := by
      have room := (fits.blocks packed).sboxColumns
      have row : (PoseidonRetainedSlots.finalRow lane).val < 150 := by
        simpa using (PoseidonRetainedSlots.finalRow lane).isLt
      have indexBound := index.isLt
      unfold sboxColumn
      have : index.val * 150 + 150 ≤ (blocksOf pieces).length * 150 := by
        have := Nat.mul_le_mul_right 150 (Nat.succ_le_of_lt indexBound)
        simpa [Nat.succ_mul] using this
      omega
    simp only [SparseLayer.evalState, sboxInterface, sboxOutputs,
      formOf_ofVar_eval _ _ bound, LogicalPlan.Layout.form_eval]
  have outputs : SparseLayer.evalState assignment
      ((sboxInterface layout (blocksOf pieces) (sboxStart pieces next)).output index) =
      ((blocksOf pieces).get index).output (LogicalPlan.sourceEnv layout assignment) := by
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

/-- The environment that an assignment decodes to: the layout's source environment, with the
dead locals of the Poseidon2 blocks completed when the blocks pack. -/
def decodeEnv (publicColumns : Nat) (pieces : List Piece) {width logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth) :
    Env :=
  if packable publicColumns pieces = true then
    complete (blocksOf pieces) (LogicalPlan.sourceEnv layout assignment)
  else LogicalPlan.sourceEnv layout assignment

/-- A satisfying assignment gives every piece's constraints in its decoded environment, which
agrees with the layout's source environment on the public columns. -/
theorem rowsZero_implies_pieces (publicColumns : Nat) (pieces : List Piece)
    (valid : ∀ piece ∈ pieces, piece.Valid) (next width : Nat)
    (fits : Fits publicColumns pieces next width) {logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment layout.one = 1)
    (rowsZero : (plan publicColumns pieces next width fits layout).RowsZero assignment) :
    (∀ piece ∈ pieces, ConstraintsHold (decodeEnv publicColumns pieces layout assignment)
        (piece.constraints (PilotData.canonicalConstraints ()))) ∧
      ∀ column, column < publicColumns →
        decodeEnv publicColumns pieces layout assignment column =
          LogicalPlan.sourceEnv layout assignment column := by
  by_cases packed : packable publicColumns pieces = true
  · unfold plan at rowsZero
    rw [dite_eq_left packed, Plan.append_rowsZero_iff] at rowsZero
    have rowsHold := (LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows layout assignment
      one).mp rowsZero.1
    unfold relationRows at rowsHold
    rw [ite_eq_left packed, R1CS.rowsHold_append] at rowsHold
    obtain ⟨holds, agree⟩ := checks_implies_pieces valid packed next _ rowsHold.1
      (outputsHold fits packed layout assignment one rowsHold.2 rowsZero.2)
    simp only [decodeEnv, packed, ↓reduceIte]
    exact ⟨holds, fun column below => agree column (packable_public packed below)⟩
  · unfold plan at rowsZero
    rw [dite_eq_right packed] at rowsZero
    have rowsHold := (LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows layout assignment
      one).mp rowsZero
    unfold relationRows at rowsHold
    rw [ite_eq_right packed] at rowsHold
    simp only [decodeEnv, packed, Bool.false_eq_true, ↓reduceIte]
    exact ⟨(PieceRows.lowerPiecesAux_sound _ _ pieces next [] valid rowsHold).2, by simp⟩

/-- The same for the constraints that `split` cut into the pieces. -/
theorem rowsZero_implies_constraints (publicColumns : Nat) (constraints : List Expr)
    (next width : Nat)
    (fits : Fits publicColumns (split (PilotData.canonicalConstraints ()) constraints) next width)
    {logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment layout.one = 1)
    (rowsZero : (plan publicColumns _ next width fits layout).RowsZero assignment) :
    ConstraintsHold (decodeEnv publicColumns (split (PilotData.canonicalConstraints ()) constraints)
        layout assignment) constraints ∧
      ∀ column, column < publicColumns →
        decodeEnv publicColumns (split (PilotData.canonicalConstraints ()) constraints) layout
          assignment column = LogicalPlan.sourceEnv layout assignment column := by
  obtain ⟨holds, agree⟩ := rowsZero_implies_pieces publicColumns _
    (valid_split _ constraints) next width fits layout assignment one rowsZero
  exact ⟨split_holds holds, agree⟩

/-! ### The step circuit's relation -/

/-- A circuit's constraints at `offset` with the decode constraints of its output digest. -/
def constraints (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat) :
    List Expr :=
  flatConstraints (Circuit.ops circuit.main offset) ++ LogicalPlan.decodeConstraints digestColumn

def pieces (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat) : List Piece :=
  split (PilotData.canonicalConstraints ()) (constraints circuit offset digestColumn)

/-- The public columns `x`. -/
abbrev publicColumns : Nat := ringDegree * PaperAlgebra.publicRingColumns

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- A satisfying assignment of the circuit's logical relation whose public input encodes
`digest` gives the circuit's spec and the output digest in the decoded environment. -/
theorem rowsZero_implies_spec (circuit : FormalCircuit) (offset firstFresh : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat}
    (fits : Fits publicColumns (pieces circuit offset digestColumn) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (plan _ _ firstFresh width fits layout).RowsZero assignment)
    (assumptions : ∀ env, circuit.assumptions offset env) :
    circuit.spec offset (decodeEnv publicColumns (pieces circuit offset digestColumn) layout
        assignment) ∧
      ∀ word, decodeEnv publicColumns (pieces circuit offset digestColumn) layout assignment
        (digestColumn word) = digest.getD word.val 0 := by
  obtain ⟨holds, agree⟩ := rowsZero_implies_constraints publicColumns
    (constraints circuit offset digestColumn) firstFresh width fits layout assignment
    (LogicalPlan.encoded_one layout assignment digest fits.rows.positive encoded) rowsZero
  let env := decodeEnv publicColumns (pieces circuit offset digestColumn) layout assignment
  have publicEq : LogicalPlan.publicInput (publicFits := publicFits) env =
      LogicalPlan.publicInput (publicFits := publicFits)
        (LogicalPlan.sourceEnv layout assignment) := by
    funext column
    exact agree column.val column.isLt
  refine ⟨?_, LogicalPlan.digest_of_decode env digestColumn digest (publicEq.trans encoded)
    (fun expression member => holds expression (List.mem_append_right _ member))⟩
  exact circuit.soundness _ offset (assumptions env)
    (holdsFlat_implies_holds _ _ fun expression member =>
      holds expression (List.mem_append_left _ member))

end NightstreamFPrime.Export.Generic.CircuitRelation
