import NightstreamFPrime.Export.Stage1.CompactRows

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CompactRowExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

def step (inputColumn : Nat → Nat) (localStart : Nat)
    (env : Env) (row : CompactTemplateRow) : Option Env :=
  let product :=
    (CompactRows.instantiateCombination inputColumn localStart row.a).eval env *
      (CompactRows.instantiateCombination inputColumn localStart row.b).eval env
  let after :=
    match row.outputLocal with
    | none => env
    | some localIndex => Env.set env (localStart + localIndex) product
  if product =
      (CompactRows.instantiateCombination inputColumn localStart row.c).eval after
  then some after else none

def run (inputColumn : Nat → Nat) (localStart : Nat) :
    Env → List CompactTemplateRow → Option Env
  | env, [] => some env
  | env, row :: rest =>
      (step inputColumn localStart env row).bind fun after =>
        run inputColumn localStart after rest

def execute (inputColumn : Nat → Nat) (localStart : Nat)
    (template : CompactRowTemplate) (env : Env) : Option Env :=
  let output := template.outputRecipe.eval (fun input => env (inputColumn input))
  let seeded := Env.set env (inputColumn template.outputInput) output
  run inputColumn localStart seeded template.rows

private theorem run_append (inputColumn : Nat → Nat) (localStart : Nat)
    (env : Env) (left right : List CompactTemplateRow) :
    run inputColumn localStart env (left ++ right) =
      (run inputColumn localStart env left).bind fun after =>
        run inputColumn localStart after right := by
  induction left generalizing env with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp only [List.cons_append, run]
      cases first : step inputColumn localStart env row with
      | none => simp []
      | some after => simpa [first] using inductionHypothesis after

theorem instantiate_abstract_self (inputCount : Nat)
    (combination : R1CS.LinearCombination) :
    CompactRows.instantiateCombination id inputCount
        (CompactRows.abstractCombination inputCount combination) =
      combination := by
  have renamed := CompactRows.instantiate_abstractCombination
    inputCount 0 id combination
  have relocateEq : CompactRows.relocate inputCount 0 id = id := by
    funext column
    simp [CompactRows.relocate]
  rw [Nat.add_zero, relocateEq] at renamed
  refine renamed.trans ?_
  cases combination
  simp [CompactRows.renameCombination, R1CS.mapCombinationColumns]

private theorem step_product (inputCount target : Nat) (env : Env)
    (left right : R1CS.LinearCombination) (targetBound : inputCount ≤ target) :
    step id inputCount env
        (CompactRows.abstractRow inputCount
          ⟨left, right, R1CS.LinearCombination.ofVar target⟩) =
      some (Env.set env target (left.eval env * right.eval env)) := by
  have outputLocal :
      CompactRows.outputLocal? inputCount (R1CS.LinearCombination.ofVar target) =
        some (target - inputCount) := by
    simp [CompactRows.outputLocal?, Rows.target?,
      R1CS.LinearCombination.ofVar, targetBound]
  have targetEq : inputCount + (target - inputCount) = target := by omega
  simp [step, CompactRows.abstractRow, outputLocal, targetEq,
    instantiate_abstract_self]

private theorem step_assertion (inputCount : Nat) (env : Env)
    (value : R1CS.LinearCombination) (valueZero : value.eval env = 0) :
    step id inputCount env
        (CompactRows.abstractRow inputCount
          ⟨value, R1CS.LinearCombination.one, R1CS.LinearCombination.zero⟩) =
      some env := by
  have noOutput :
      CompactRows.outputLocal? inputCount R1CS.LinearCombination.zero = none := by
    simp [CompactRows.outputLocal?, Rows.target?, R1CS.LinearCombination.zero]
  simp [step, CompactRows.abstractRow, noOutput, instantiate_abstract_self, valueZero]

theorem run_lowerExpression (inputCount : Nat) (env : Env)
    (expression : Expr) (start : Nat) (startBound : inputCount ≤ start) :
    run id inputCount env
        ((R1CS.lowerExpression expression start).rows.map
          (CompactRows.abstractRow inputCount)) =
      some (R1CS.executeExpression env expression start) := by
  induction expression generalizing env start with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp only [R1CS.lowerExpression, R1CS.lowerExpression_next, List.map_append]
      rw [run_append, leftIH env start startBound]
      exact rightIH (R1CS.executeExpression env left start)
        (start + R1CS.mulCount left) (by omega)
  | mul left right leftIH rightIH =>
      simp only [R1CS.lowerExpression, R1CS.lowerExpression_next,
        List.map_append, List.map_singleton]
      rw [run_append, run_append, leftIH env start startBound]
      simp only [Option.bind_some]
      rw [rightIH (R1CS.executeExpression env left start)
        (start + R1CS.mulCount left) (by omega)]
      simp only [Option.bind_some, run]
      rw [step_product inputCount
        (start + R1CS.mulCount left + R1CS.mulCount right)
        _ _ _ (by omega)]; rfl

theorem run_lowerGenericConstraint (inputCount : Nat) (env : Env)
    (expression : Expr) (start : Nat) (startBound : inputCount ≤ start)
    (scope : expression.VarsBelow start) (logical : expression.eval env = 0) :
    run id inputCount env
        ((R1CS.lowerGenericConstraint expression start).rows.map
          (CompactRows.abstractRow inputCount)) =
      some (R1CS.executeExpression env expression start) := by
  let completed := R1CS.executeExpression env expression start
  have agrees := R1CS.executeExpression_agreesOutside env expression start
  have completedZero : expression.eval completed = 0 := by
    rw [expression.eval_eq_of_agree_below start completed env scope
      (fun index below => agrees index (Or.inl below))]
    exact logical
  have valueZero : (R1CS.lowerExpression expression start).value.eval completed = 0 :=
    (R1CS.lowerExpression_sound completed expression start
      (R1CS.executeExpression_holds_rows env expression start scope)).trans completedZero
  simp only [R1CS.lowerGenericConstraint, List.map_append, List.map_singleton]
  rw [run_append, run_lowerExpression inputCount env expression start startBound]
  simp only [Option.bind_some, run]
  rw [step_assertion inputCount completed
    (R1CS.lowerExpression expression start).value valueZero]; rfl

end NightstreamFPrime.Export.Stage1.CompactRowExecution
