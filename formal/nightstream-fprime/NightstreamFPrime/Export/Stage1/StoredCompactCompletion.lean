import NightstreamFPrime.Export.Stage1.StoredCompactRowExecution
import NightstreamFPrime.Export.Stage1.CompactOutputExecution
import NightstreamFPrime.Export.Stage1.CompactRowRelocation

/-!
Compose stored execution, physical row relocation and the canonical lowerers
for the two compact-template constructors. All geometry premises remain explicit.
The reference environment is the pullback after the physical output write.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredCompactCompletion

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open StoredWitnessExecution (asEnv)
open CompactRowRelocation (pullback)

/-- A physical output write preserves every input used by the recipe. Other
input aliases are unrestricted. Reuse the existing normalized output law. -/
private theorem logical_after_physical_output
    (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
    (env : Env) (recipe : Expr) (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput)
    (outputDistinct : ∀ input, input < outputInput →
      inputColumn input ≠ inputColumn outputInput) :
    (Expr.var outputInput - recipe).eval
      (pullback inputCount localStart inputColumn
        (Env.set env (inputColumn outputInput)
          (recipe.eval (fun input => env (inputColumn input))))) = 0 := by
  let inputEnv : Env := fun input => env (inputColumn input)
  let value := recipe.eval inputEnv
  let physical := Env.set env (inputColumn outputInput) value
  let normalized := Env.set inputEnv outputInput value
  have agrees : ∀ index, index < outputInput + 1 →
      pullback inputCount localStart inputColumn physical index = normalized index := by
    intro index below
    have input : index < inputCount := by omega
    dsimp only [pullback]
    rw [CompactRows.relocate_input inputCount (localStart - inputCount)
      inputColumn index input]
    by_cases same : index = outputInput
    · subst index
      simp only [physical, normalized, Env.set_self]
    · have earlier : index < outputInput := by omega
      have physicalNe := outputDistinct index earlier
      simp only [physical, normalized, Env.set, same, physicalNe, if_false, inputEnv]
  have scope : (Expr.var outputInput - recipe).VarsBelow (outputInput + 1) :=
    Expr.VarsBelow.sub _ _ _ (Nat.lt_succ_self _)
      (recipeScope.mono recipe (Nat.le_succ _))
  change (Expr.var outputInput - recipe).eval
    (pullback inputCount localStart inputColumn physical) = 0
  rw [(Expr.var outputInput - recipe).eval_eq_of_agree_below
    (outputInput + 1) (pullback inputCount localStart inputColumn physical)
    normalized scope agrees]
  exact CompactOutputExecution.logical_after_output inputEnv outputInput recipe recipeScope

/-- Local guards follow from the canonical lowerer's C-column scope. -/
private theorem abstract_localBounds (inputCount count : Nat) (rows : List R1CS.Row)
    (scope : ∀ row ∈ rows, row.c.VarsBelow (inputCount + count)) :
    ∀ row ∈ rows.map (CompactRows.abstractRow inputCount), ∀ index,
      row.outputLocal = some index → index < count := by
  intro row member index found
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact CompactRowRelocation.outputLocal_bound inputCount count source.c
    (scope source sourceMember) index found

private theorem run_lowerGenericConstraint (inputCount localStart : Nat)
    (inputColumn : Nat → Nat) (env : Env) (expression : Expr)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + R1CS.mulCount expression ≤ inputColumn input)
    (scope : expression.VarsBelow inputCount)
    (logical : expression.eval (pullback inputCount localStart inputColumn env) = 0) :
    (CompactRowExecution.run inputColumn localStart env
        ((R1CS.lowerGenericConstraint expression inputCount).rows.map
          (CompactRows.abstractRow inputCount))).map
            (pullback inputCount localStart inputColumn) =
      some (R1CS.executeExpression
        (pullback inputCount localStart inputColumn env) expression inputCount) := by
  rw [CompactRowRelocation.run_relocate inputCount localStart
    (R1CS.mulCount expression) inputColumn env
    (R1CS.lowerGenericConstraint expression inputCount).rows localBound inputsOutside
    (fun row member =>
      (R1CS.lowerGenericConstraint_rows_varsBelow
        expression inputCount scope row member).2.2)]
  exact CompactRowExecution.run_lowerGenericConstraint inputCount
    (pullback inputCount localStart inputColumn env) expression inputCount
    (Nat.le_refl _) scope logical

/-- Stored execution of a generic canonical compact template has exactly the
existing expression executor's result after the initial physical output write. -/
theorem execute_compactTemplate (inputCount outputInput localStart : Nat)
    (inputColumn : Nat → Nat) (recipe : Expr) (values : Array F)
    (outputFits : inputColumn outputInput < values.size)
    (localFits : localStart + R1CS.mulCount (Expr.var outputInput - recipe) ≤ values.size)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + R1CS.mulCount (Expr.var outputInput - recipe) ≤ inputColumn input)
    (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput)
    (outputDistinct : ∀ input, input < outputInput →
      inputColumn input ≠ inputColumn outputInput) :
    (StoredCompactRowExecution.execute inputColumn localStart
        (CompactRows.compactTemplate inputCount outputInput recipe) values).map
          (pullback inputCount localStart inputColumn ∘ asEnv) =
      some (R1CS.executeExpression
        (pullback inputCount localStart inputColumn
          (Env.set (asEnv values) (inputColumn outputInput)
            (recipe.eval (fun input => asEnv values (inputColumn input)))))
        (Expr.var outputInput - recipe) inputCount) := by
  let expression := Expr.var outputInput - recipe
  let template := CompactRows.compactTemplate inputCount outputInput recipe
  let seeded := Env.set (asEnv values) (inputColumn outputInput)
    (recipe.eval (fun input => asEnv values (inputColumn input)))
  have scope : expression.VarsBelow inputCount :=
    Expr.VarsBelow.sub _ _ _ outputBelow
      (recipeScope.mono recipe (Nat.le_of_lt outputBelow))
  have localBounds : ∀ row ∈ template.rows, ∀ index,
      row.outputLocal = some index → index < template.localColumnCount := by
    exact abstract_localBounds inputCount (R1CS.mulCount expression)
      (R1CS.lowerGenericConstraint expression inputCount).rows
      (fun row member =>
        (R1CS.lowerGenericConstraint_rows_varsBelow
          expression inputCount scope row member).2.2)
  have stored := StoredCompactRowExecution.execute_eq inputColumn localStart
    template values outputFits localFits localBounds
  have logical : expression.eval (pullback inputCount localStart inputColumn seeded) = 0 :=
    logical_after_physical_output inputCount outputInput localStart inputColumn
      (asEnv values) recipe outputBelow recipeScope outputDistinct
  have rows :
      (CompactRowExecution.execute inputColumn localStart template (asEnv values)).map
          (pullback inputCount localStart inputColumn) =
        some (R1CS.executeExpression
          (pullback inputCount localStart inputColumn seeded) expression inputCount) := by
    exact run_lowerGenericConstraint inputCount localStart inputColumn seeded
      expression localBound inputsOutside scope logical
  calc
    (StoredCompactRowExecution.execute inputColumn localStart template values).map
        (pullback inputCount localStart inputColumn ∘ asEnv) =
      ((StoredCompactRowExecution.execute inputColumn localStart template values).map asEnv).map
        (pullback inputCount localStart inputColumn) := by rw [Option.map_map]
    _ = (CompactRowExecution.execute inputColumn localStart template (asEnv values)).map
        (pullback inputCount localStart inputColumn) :=
      congrArg (Option.map (pullback inputCount localStart inputColumn)) stored
    _ = _ := rows

/-- Stored execution keeps the optimized directConstraint branch and its generic
fallback exactly as selected by the existing canonical constraint lowerer. -/
theorem execute_compactConstraintTemplate (inputCount outputInput localStart : Nat)
    (inputColumn : Nat → Nat) (recipe : Expr) (values : Array F)
    (outputFits : inputColumn outputInput < values.size)
    (localFits : localStart +
      R1CS.constraintFreshCount (Expr.var outputInput - recipe) ≤ values.size)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨ localStart +
        R1CS.constraintFreshCount (Expr.var outputInput - recipe) ≤ inputColumn input)
    (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput)
    (outputDistinct : ∀ input, input < outputInput →
      inputColumn input ≠ inputColumn outputInput) :
    (StoredCompactRowExecution.execute inputColumn localStart
        (CompactRows.compactConstraintTemplate inputCount outputInput recipe) values).map
          (pullback inputCount localStart inputColumn ∘ asEnv) =
      some (R1CS.executeConstraint
        (pullback inputCount localStart inputColumn
          (Env.set (asEnv values) (inputColumn outputInput)
            (recipe.eval (fun input => asEnv values (inputColumn input)))))
        (Expr.var outputInput - recipe) inputCount) := by
  let expression := Expr.var outputInput - recipe
  let template := CompactRows.compactConstraintTemplate inputCount outputInput recipe
  let seeded := Env.set (asEnv values) (inputColumn outputInput)
    (recipe.eval (fun input => asEnv values (inputColumn input)))
  have scope : expression.VarsBelow inputCount :=
    Expr.VarsBelow.sub _ _ _ outputBelow
      (recipeScope.mono recipe (Nat.le_of_lt outputBelow))
  have localBounds : ∀ row ∈ template.rows, ∀ index,
      row.outputLocal = some index → index < template.localColumnCount := by
    exact abstract_localBounds inputCount (R1CS.constraintFreshCount expression)
      (R1CS.lowerConstraint expression inputCount).rows
      (fun row member =>
        (R1CS.lowerConstraint_rows_varsBelow expression inputCount scope row member).2.2)
  have stored := StoredCompactRowExecution.execute_eq inputColumn localStart
    template values outputFits localFits localBounds
  have logical : expression.eval (pullback inputCount localStart inputColumn seeded) = 0 :=
    logical_after_physical_output inputCount outputInput localStart inputColumn
      (asEnv values) recipe outputBelow recipeScope outputDistinct
  have rows :
      (CompactRowExecution.execute inputColumn localStart template (asEnv values)).map
          (pullback inputCount localStart inputColumn) =
        some (R1CS.executeConstraint
          (pullback inputCount localStart inputColumn seeded) expression inputCount) := by
    exact CompactRowRelocation.run_lowerConstraint inputCount localStart inputColumn
      seeded expression localBound inputsOutside scope logical
  calc
    (StoredCompactRowExecution.execute inputColumn localStart template values).map
        (pullback inputCount localStart inputColumn ∘ asEnv) =
      ((StoredCompactRowExecution.execute inputColumn localStart template values).map asEnv).map
        (pullback inputCount localStart inputColumn) := by rw [Option.map_map]
    _ = (CompactRowExecution.execute inputColumn localStart template (asEnv values)).map
        (pullback inputCount localStart inputColumn) :=
      congrArg (Option.map (pullback inputCount localStart inputColumn)) stored
    _ = _ := rows

end NightstreamFPrime.Export.Stage1.StoredCompactCompletion
