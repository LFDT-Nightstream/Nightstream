import NightstreamFPrime.Export.Stage1.CompactRowExecution

/-!
The optimized constraint branch of the compact row executor. Direct rows
read only input columns and perform no local write. The generic fallback
reuses CompactRowExecution.run_lowerGenericConstraint unchanged.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CompactConstraintExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1.CompactRowExecution

private theorem outputLocal_none_of_varsBelow (inputCount : Nat)
    (combination : R1CS.LinearCombination)
    (scope : combination.VarsBelow inputCount) :
    CompactRows.outputLocal? inputCount combination = none := by
  unfold CompactRows.outputLocal?
  cases found : Rows.target? combination with
  | none => rfl
  | some target =>
      have selected := Rows.target?_eq_some found
      have targetBelow : target < inputCount :=
        scope (target, 1) (by
          rw [selected]
          simp [R1CS.LinearCombination.ofVar])
      simp [Nat.not_le_of_gt targetBelow]

private theorem step_input_row (inputCount : Nat) (env : Env)
    (row : R1CS.Row) (scope : row.c.VarsBelow inputCount)
    (holds : row.Holds env) :
    step id inputCount env (CompactRows.abstractRow inputCount row) = some env := by
  have noOutput := outputLocal_none_of_varsBelow inputCount row.c scope
  have equation : row.a.eval env * row.b.eval env = row.c.eval env := holds
  simp [step, CompactRows.abstractRow, noOutput, instantiate_abstract_self, equation]

/-- Keep the exact optimized/generic branch selected by directConstraint.
Input scope excludes a local target in the direct row; logical zero supplies
its existing completeness proof. No new direct-row recognizer is used. -/
theorem run_lowerConstraint (inputCount : Nat) (env : Env)
    (expression : Expr) (start : Nat) (startBound : inputCount ≤ start)
    (scope : expression.VarsBelow inputCount) (logical : expression.eval env = 0) :
    run id inputCount env
        ((R1CS.lowerConstraint expression start).rows.map
          (CompactRows.abstractRow inputCount)) =
      some (R1CS.executeConstraint env expression start) := by
  cases found : R1CS.directConstraint expression with
  | none =>
      simpa [R1CS.lowerConstraint, R1CS.executeConstraint, found] using
        (run_lowerGenericConstraint inputCount env expression start startBound
          (Expr.VarsBelow.mono expression scope startBound) logical)
  | some result =>
      have sourceScope := R1CS.lowerConstraint_rows_varsBelow expression
        inputCount scope result.row (by simp [R1CS.lowerConstraint, found])
      have rowScope : result.row.VarsBelow inputCount := by
        simpa [R1CS.constraintFreshCount, found] using sourceScope
      have passed := step_input_row inputCount env result.row rowScope.2.2
        (result.complete env logical)
      simp [R1CS.lowerConstraint, R1CS.executeConstraint, found, run, passed]

/-- Exact canonical template connection with its output index and recipe
scope explicit. The environment is the state on entry to the row phase;
establishing logical zero after the initial output write is a separate link. -/
theorem run_compactConstraintTemplate (inputCount outputInput : Nat)
    (recipe : Expr) (env : Env) (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow inputCount)
    (logical : (Expr.var outputInput - recipe).eval env = 0) :
    run id inputCount env
        (CompactRows.compactConstraintTemplate inputCount outputInput recipe).rows =
      some (R1CS.executeConstraint env (Expr.var outputInput - recipe) inputCount) := by
  exact run_lowerConstraint inputCount env (Expr.var outputInput - recipe)
    inputCount (Nat.le_refl _)
    (Expr.VarsBelow.sub _ _ _ outputBelow recipeScope) logical

end NightstreamFPrime.Export.Stage1.CompactConstraintExecution
