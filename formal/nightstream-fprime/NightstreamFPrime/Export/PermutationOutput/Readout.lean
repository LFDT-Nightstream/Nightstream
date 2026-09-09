import NightstreamFPrime.Export.PermutationOutput

/-!
Owns readout of a contiguous family of canonical Poseidon2 permutations.
Address proofs use symbolic family geometry, so their cost does not depend
on a concrete package layout. No row acceptance is assumed by the readout.
-/

namespace NightstreamFPrime.Export.PermutationOutput.Readout

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout.ProductionRelation

def witnessStart (start : Nat) {count : Nat} (index : Fin count) : Nat :=
  start + index.val * 592

def outputColumn (start : Nat) {count : Nat} (index : Fin count) (lane : Fin 8) : Nat :=
  witnessStart start index + 584 + lane.val

def sboxColumn (start : Nat) {count : Nat} (index : Fin count) (lane : Fin 8) : Nat :=
  witnessStart start index +
    (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val

def decode (start count column : Nat) : Option (Fin count × Fin 8) :=
  if _lower : start + 584 ≤ column then
    let offset := column - (start + 584)
    if invocationBound : offset / 592 < count then
      if laneBound : offset % 592 < 8 then
        some (⟨offset / 592, invocationBound⟩, ⟨offset % 592, laneBound⟩)
      else none
    else none
  else none

theorem decode_outputColumn (start : Nat) {count : Nat}
    (index : Fin count) (lane : Fin 8) :
    decode start count (outputColumn start index lane) = some (index, lane) := by
  have lower : start + 584 ≤ outputColumn start index lane := by
    unfold outputColumn witnessStart
    omega
  have offset : outputColumn start index lane - (start + 584) =
      index.val * 592 + lane.val := by
    unfold outputColumn witnessStart
    omega
  have laneBound := lane.isLt
  have quotient : (index.val * 592 + lane.val) / 592 = index.val := by omega
  have remainder : (index.val * 592 + lane.val) % 592 = lane.val := by omega
  unfold decode
  rw [dif_pos lower]
  simp only [offset, quotient, remainder, dif_pos index.isLt, dif_pos lane.isLt]

theorem decode_source (start : Nat) {count column : Nat}
    {index : Fin count} {lane : Fin 8}
    (found : decode start count column = some (index, lane)) :
    column = outputColumn start index lane := by
  unfold decode at found
  split at found
  · rename_i lower
    dsimp only at found
    split at found
    · split at found
      · have selected := Option.some.inj found
        have indexEq := congrArg (fun pair : Fin count × Fin 8 => pair.1.val) selected
        have laneEq := congrArg (fun pair : Fin count × Fin 8 => pair.2.val) selected
        dsimp only at indexEq laneEq
        have divmod := Nat.mod_add_div (column - (start + 584)) 592
        unfold outputColumn witnessStart
        omega
      · cases found
    · cases found
  · cases found

private theorem sboxOffset_bounds (lane : Fin 8) :
    555 ≤ (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val ∧
    (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val ≤ 583 := by
  fin_cases lane <;> decide

theorem sboxColumn_lt_end (start : Nat) {count : Nat}
    (index : Fin count) (lane : Fin 8) :
    sboxColumn start index lane < start + count * 592 := by
  have slotBound := sboxOffset_bounds lane
  have indexBound := index.isLt
  unfold sboxColumn witnessStart
  omega

theorem decode_sboxColumn (start : Nat) {count : Nat}
    (index : Fin count) (lane : Fin 8) :
    decode start count (sboxColumn start index lane) = none := by
  cases found : decode start count (sboxColumn start index lane) with
  | none => rfl
  | some selected =>
      rcases selected with ⟨other, outputLane⟩
      have same := decode_source start found
      have bounded := sboxOffset_bounds lane
      have outputBound := outputLane.isLt
      unfold sboxColumn outputColumn witnessStart at same
      omega

def env (start count : Nat) (source : Env) : Env := fun column =>
  match decode start count column with
  | none => source column
  | some (index, lane) =>
      Layer.externalF (fun selected => source (sboxColumn start index selected)) lane

/-- Readout at one column depends only on that source column and the retained
final S-boxes. This transports source agreement through the computed view. -/
theorem env_congr_at (start count : Nat) (left right : Env) (column : Nat)
    (atColumn : left column = right column)
    (atSboxes : ∀ (index : Fin count) (lane : Fin 8),
      left (sboxColumn start index lane) = right (sboxColumn start index lane)) :
    env start count left column = env start count right column := by
  cases found : decode start count column with
  | none => simpa only [env, found] using atColumn
  | some selected =>
      rcases selected with ⟨index, lane⟩
      simp only [env, found]
      apply congrArg (fun state => Layer.externalF state lane)
      funext selected
      exact atSboxes index selected

theorem env_outputColumn (start : Nat) {count : Nat}
    (source : Env) (index : Fin count) (lane : Fin 8) :
    env start count source (outputColumn start index lane) =
      Layer.externalF (fun selected => source (sboxColumn start index selected)) lane := by
  simp only [env, decode_outputColumn]

theorem env_of_decode_none (start count : Nat) (source : Env) (column : Nat)
    (outside : decode start count column = none) :
    env start count source column = source column := by
  simp only [env, outside]

theorem env_sboxColumn (start : Nat) {count : Nat}
    (source : Env) (index : Fin count) (lane : Fin 8) :
    env start count source (sboxColumn start index lane) =
      source (sboxColumn start index lane) := by
  exact env_of_decode_none start count source _ (decode_sboxColumn start index lane)

theorem env_idempotent (start count : Nat) (source : Env) :
    env start count (env start count source) = env start count source := by
  funext column
  cases found : decode start count column with
  | none => simp only [env, found]
  | some selected =>
      rcases selected with ⟨index, lane⟩
      rw [decode_source start found, env_outputColumn, env_outputColumn]
      apply congrArg (fun state => Layer.externalF state lane)
      funext selected
      exact env_sboxColumn start source index selected

/-- The exported expression for one source column uses the same readout as
the verifier environment. No additional variable or copy equation is added. -/
def variableExpr (start count column : Nat) : Expr :=
  match decode start count column with
  | none => .var column
  | some (index, lane) =>
      Layer.externalE (fun selected => .var (sboxColumn start index selected)) lane

theorem variableExpr_eval (start count column : Nat) (source : Env) :
    (variableExpr start count column).eval source = env start count source column := by
  cases found : decode start count column with
  | none => simp only [variableExpr, env, found, Expr.eval_var]
  | some selected =>
      rcases selected with ⟨index, lane⟩
      simp only [variableExpr, env, found, Layer.eval_externalE]
      rfl

/-- Substitute computed readouts structurally in an exported arithmetic
expression. The proof follows the expression, not a materialized package. -/
def rewriteExpr (start count : Nat) : Expr → Expr
  | .var column => variableExpr start count column
  | .const value => .const value
  | .add left right => .add (rewriteExpr start count left) (rewriteExpr start count right)
  | .mul left right => .mul (rewriteExpr start count left) (rewriteExpr start count right)

theorem rewriteExpr_eval (start count : Nat) (source : Env) (expression : Expr) :
    (rewriteExpr start count expression).eval source =
      expression.eval (env start count source) := by
  induction expression with
  | var column => exact variableExpr_eval start count column source
  | const value => rfl
  | add left right leftIH rightIH => simp only [rewriteExpr, Expr.eval_add, leftIH, rightIH]
  | mul left right leftIH rightIH => simp only [rewriteExpr, Expr.eval_mul, leftIH, rightIH]

end NightstreamFPrime.Export.PermutationOutput.Readout
