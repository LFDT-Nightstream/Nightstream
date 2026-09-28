import NightstreamFPrime.Export.Stage1.CompactConstraintExecution

/-!
Relate physical compact row execution to the normalized row executor.
Input columns may alias each other. They must be outside the fresh interval,
and every local write must remain inside that interval. The output-recipe
write and array bounds are separate obligations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CompactRowRelocation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1.CompactRowExecution

def pullback (inputCount localStart : Nat) (inputColumn : Nat → Nat)
    (env : Env) : Env :=
  fun index =>
    env (CompactRows.relocate inputCount (localStart - inputCount) inputColumn index)

/-- Only writes inside the fresh interval commute with this pullback.
No injectivity assumption is made about the input-column map. -/
private theorem pullback_setLocal (inputCount localStart count : Nat)
    (inputColumn : Nat → Nat) (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨ localStart + count ≤ inputColumn input)
    (env : Env) (localIndex : Nat) (indexBound : localIndex < count) (value : F) :
    pullback inputCount localStart inputColumn
        (Env.set env (localStart + localIndex) value) =
      Env.set (pullback inputCount localStart inputColumn env)
        (inputCount + localIndex) value := by
  funext index
  simp only [pullback, Env.set]
  by_cases input : index < inputCount
  · rw [CompactRows.relocate_input inputCount (localStart - inputCount)
      inputColumn index input]
    have physicalNe : inputColumn index ≠ localStart + localIndex := by
      rcases inputsOutside index input with before | after <;> omega
    have normalizedNe : index ≠ inputCount + localIndex := by omega
    simp only [physicalNe, normalizedNe, if_false]
  · rw [CompactRows.relocate_local inputCount (localStart - inputCount)
      inputColumn index (Nat.le_of_not_gt input)]
    have sameTarget :
        index + (localStart - inputCount) = localStart + localIndex ↔
          index = inputCount + localIndex := by omega
    simp only [sameTarget]

private theorem abstract_eval (inputCount localStart : Nat)
    (inputColumn : Nat → Nat) (localBound : inputCount ≤ localStart)
    (combination : R1CS.LinearCombination) (env : Env) :
    (CompactRows.instantiateCombination inputColumn localStart
      (CompactRows.abstractCombination inputCount combination)).eval env =
        combination.eval (pullback inputCount localStart inputColumn env) := by
  have startEq : inputCount + (localStart - inputCount) = localStart := by omega
  have mapped := CompactRows.instantiate_abstractCombination inputCount
    (localStart - inputCount) inputColumn combination
  rw [startEq] at mapped
  rw [mapped, CompactRows.renameCombination_eval] <;> rfl

theorem outputLocal_bound (inputCount count : Nat)
    (combination : R1CS.LinearCombination)
    (scope : combination.VarsBelow (inputCount + count))
    (localIndex : Nat)
    (found : CompactRows.outputLocal? inputCount combination = some localIndex) :
    localIndex < count := by
  cases recognized : Rows.target? combination with
  | none => simp [CompactRows.outputLocal?, recognized] at found
  | some target =>
      have selected := Rows.target?_eq_some recognized
      have targetBelow : target < inputCount + count :=
        scope (target, 1) (by
          rw [selected]
          simp [R1CS.LinearCombination.ofVar])
      by_cases targetLocal : inputCount ≤ target
      · simp only [CompactRows.outputLocal?, recognized, if_pos targetLocal,
          Option.some.injEq] at found
        omega
      · simp [CompactRows.outputLocal?, recognized, targetLocal] at found

private theorem step_relocate (inputCount localStart count : Nat)
    (inputColumn : Nat → Nat) (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨ localStart + count ≤ inputColumn input)
    (env : Env) (row : R1CS.Row)
    (scope : row.c.VarsBelow (inputCount + count)) :
    (step inputColumn localStart env (CompactRows.abstractRow inputCount row)).map
        (pullback inputCount localStart inputColumn) =
      step id inputCount (pullback inputCount localStart inputColumn env)
        (CompactRows.abstractRow inputCount row) := by
  cases target : CompactRows.outputLocal? inputCount row.c with
  | none =>
      simp only [step, CompactRows.abstractRow, target,
        abstract_eval inputCount localStart inputColumn localBound,
        instantiate_abstract_self]
      split_ifs <;> rfl
  | some localIndex =>
      let product :=
        row.a.eval (pullback inputCount localStart inputColumn env) *
          row.b.eval (pullback inputCount localStart inputColumn env)
      have wrote := pullback_setLocal inputCount localStart count inputColumn
        localBound inputsOutside env localIndex
        (outputLocal_bound inputCount count row.c scope localIndex target) product
      simp only [step, CompactRows.abstractRow, target,
        abstract_eval inputCount localStart inputColumn localBound,
        instantiate_abstract_self]
      dsimp only [product] at wrote
      simp only [wrote]
      split_ifs <;> simp only [Option.map_some, Option.map_none, wrote]

/-- Complete success/failure correspondence for an ordered list of abstract
rows. A bound on C alone bounds every recognized local target; the exact
combination transport handles all input and local reads. -/
theorem run_relocate (inputCount localStart count : Nat)
    (inputColumn : Nat → Nat) (env : Env) (rows : List R1CS.Row)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨ localStart + count ≤ inputColumn input)
    (scope : ∀ row ∈ rows, row.c.VarsBelow (inputCount + count)) :
    (run inputColumn localStart env
        (rows.map (CompactRows.abstractRow inputCount))).map
          (pullback inputCount localStart inputColumn) =
      run id inputCount (pullback inputCount localStart inputColumn env)
        (rows.map (CompactRows.abstractRow inputCount)) := by
  induction rows generalizing env with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      have first := step_relocate inputCount localStart count inputColumn
        localBound inputsOutside env row (scope row (by simp))
      have tailScope : ∀ current ∈ rest,
          current.c.VarsBelow (inputCount + count) := by
        intro current member
        exact scope current (by simp [member])
      cases result : step inputColumn localStart env
          (CompactRows.abstractRow inputCount row) with
      | none =>
          have normalized :
              step id inputCount (pullback inputCount localStart inputColumn env)
                (CompactRows.abstractRow inputCount row) = none := by
            simpa [result] using first.symm
          simp [List.map_cons, run, result, normalized]
      | some after =>
          have normalized :
              step id inputCount (pullback inputCount localStart inputColumn env)
                  (CompactRows.abstractRow inputCount row) =
                some (pullback inputCount localStart inputColumn after) := by
            simpa [result] using first.symm
          simpa only [List.map_cons, run, result, normalized, Option.bind_some] using
            inductionHypothesis after tailScope

/-- Canonical optimized/generic row bounds come from the existing lowerer.
The row-phase logical premise is evaluated on the exact physical pullback. -/
theorem run_lowerConstraint (inputCount localStart : Nat)
    (inputColumn : Nat → Nat) (env : Env) (expression : Expr)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + R1CS.constraintFreshCount expression ≤ inputColumn input)
    (scope : expression.VarsBelow inputCount)
    (logical : expression.eval (pullback inputCount localStart inputColumn env) = 0) :
    (run inputColumn localStart env
        ((R1CS.lowerConstraint expression inputCount).rows.map
          (CompactRows.abstractRow inputCount))).map
            (pullback inputCount localStart inputColumn) =
      some (R1CS.executeConstraint
        (pullback inputCount localStart inputColumn env) expression inputCount) := by
  rw [run_relocate inputCount localStart
    (R1CS.constraintFreshCount expression) inputColumn env
    (R1CS.lowerConstraint expression inputCount).rows localBound inputsOutside
    (fun row member =>
      (R1CS.lowerConstraint_rows_varsBelow expression inputCount scope row member).2.2)]
  exact CompactConstraintExecution.run_lowerConstraint inputCount
    (pullback inputCount localStart inputColumn env) expression inputCount
    (Nat.le_refl _) scope logical

end NightstreamFPrime.Export.Stage1.CompactRowRelocation
