import NightstreamFPrime.Circuit.Execution
import NightstreamFPrime.Export.Generic.RelationCompleteness

/-! The native fills of a step package's witness program (`StepPackage.nativeRanges`,
`SECURITY_MODEL.md`, trust gap 5). Rust fills a block's 16 output locals with the permutation of
its lanes (`fillBlock`) and the outputs of a ring piece with the prior plus the challenge times
the value (`fillRing`), in place of the circuit's batches of the range. When the piece's rows
hold after those batches, the fill gives their values off the block's dead locals
(`agreeOff_block`, `agreeOff_ring`). -/

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
