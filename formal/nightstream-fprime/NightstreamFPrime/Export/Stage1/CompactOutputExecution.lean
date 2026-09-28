import NightstreamFPrime.Export.Stage1.CompactConstraintExecution

/-! Connect the initial output write to both canonical compact row families.
Recipes read only earlier normalized inputs, so the output write preserves them. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CompactOutputExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open CompactRowExecution

theorem logical_after_output (env : Env) (outputInput : Nat) (recipe : Expr)
    (scope : recipe.VarsBelow outputInput) :
    (Expr.var outputInput - recipe).eval
      (Env.set env outputInput (recipe.eval env)) = 0 := by
  have preserved : recipe.eval (Env.set env outputInput (recipe.eval env)) =
      recipe.eval env := by
    apply recipe.eval_eq_of_agree_below outputInput _ env scope
    intro index below
    exact Env.set_of_ne env outputInput index _ (Nat.ne_of_lt below)
  simp [Expr.eval, preserved]

theorem execute_compactTemplate (inputCount outputInput : Nat)
    (recipe : Expr) (env : Env) (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput) :
    execute id inputCount (CompactRows.compactTemplate inputCount outputInput recipe) env =
      some (R1CS.executeExpression (Env.set env outputInput (recipe.eval env))
        (Expr.var outputInput - recipe) inputCount) := by
  have scope := Expr.VarsBelow.sub (.var outputInput) recipe inputCount outputBelow
    (recipeScope.mono recipe (Nat.le_of_lt outputBelow))
  exact run_lowerGenericConstraint inputCount
    (Env.set env outputInput (recipe.eval env)) (Expr.var outputInput - recipe)
    inputCount (Nat.le_refl _) scope (logical_after_output env outputInput recipe recipeScope)

theorem execute_compactConstraintTemplate (inputCount outputInput : Nat)
    (recipe : Expr) (env : Env) (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput) :
    execute id inputCount
        (CompactRows.compactConstraintTemplate inputCount outputInput recipe) env =
      some (R1CS.executeConstraint (Env.set env outputInput (recipe.eval env))
        (Expr.var outputInput - recipe) inputCount) := by
  exact CompactConstraintExecution.run_compactConstraintTemplate inputCount outputInput recipe
    (Env.set env outputInput (recipe.eval env)) outputBelow
    (recipeScope.mono recipe (Nat.le_of_lt outputBelow))
    (logical_after_output env outputInput recipe recipeScope)

end NightstreamFPrime.Export.Stage1.CompactOutputExecution
