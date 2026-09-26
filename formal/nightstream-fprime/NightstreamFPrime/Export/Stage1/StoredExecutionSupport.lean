import NightstreamFPrime.Export.Stage1.StoredInstructionExecution
import NightstreamFPrime.Export.Stage1.StoredPermutationExecution
import NightstreamFPrime.Export.Stage1.StoredCompactRowExecution
import NightstreamFPrime.Layout.R1CS.Support

/-!
Read support for the existing array interpreters. Agreement includes equal
storage sizes, so bounds checks and out-of-bounds reads remain identical.
No canonical schedule or removable scratch interval is assumed here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredExecutionSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open StoredWitnessExecution (asEnv write)

def Agree (allowed : Nat → Prop) (left right : Array F) : Prop :=
  left.size = right.size ∧ ∀ column, allowed column → asEnv left column = asEnv right column

theorem write_agree (allowed : Nat → Prop) (left right : Array F)
    (target : Nat) (value : F) (agree : Agree allowed left right) :
    Agree allowed (write left target value) (write right target value) := by
  refine ⟨by simpa only [StoredWitnessExecution.write_size] using agree.1, ?_⟩
  intro column supported
  by_cases same : target = column
  · subst column
    simp only [asEnv, write, Array.getElem?_setIfInBounds_self, agree.1]
  · simpa only [asEnv, write, Array.getElem?_setIfInBounds_ne same] using
      agree.2 column supported

theorem recipes_agree (allowed : Nat → Prop) (left right : Array F)
    (start : Nat) (recipes : List Expr)
    (support : ∀ recipe ∈ recipes, recipe.VarsSatisfy allowed)
    (agree : Agree allowed left right) :
    Agree allowed (StoredWitnessExecution.executeRecipes left start recipes)
      (StoredWitnessExecution.executeRecipes right start recipes) := by
  induction recipes generalizing left right start with
  | nil => exact agree
  | cons recipe rest ih =>
      have valueEq := recipe.eval_eq_of_agree_satisfy allowed (asEnv left) (asEnv right)
        (support recipe (by simp)) agree.2
      simp only [StoredWitnessExecution.executeRecipes, valueEq]
      apply ih _ _ (start + 1) (fun item member => support item (by simp [member]))
      exact write_agree allowed left right start _ agree

theorem hint_eval_eq (hint : Hint) (allowed : Nat → Prop) (left right : Env)
    (support : hint.source.VarsSatisfy allowed)
    (agree : ∀ column, allowed column → left column = right column) :
    hint.eval left = hint.eval right := by
  have sourceEq := hint.source.eval_eq_of_agree_satisfy allowed left right support agree
  cases hint <;> simp only [Hint.source] at sourceEq <;> simp only [Hint.eval, sourceEq]

theorem hints_agree (allowed : Nat → Prop) (left right : Array F)
    (start : Nat) (hints : List Hint)
    (support : ∀ hint ∈ hints, hint.source.VarsSatisfy allowed)
    (agree : Agree allowed left right) :
    Agree allowed (StoredWitnessExecution.executeHints left start hints)
      (StoredWitnessExecution.executeHints right start hints) := by
  induction hints generalizing left right start with
  | nil => exact agree
  | cons hint rest ih =>
      have valueEq := hint_eval_eq hint allowed (asEnv left) (asEnv right)
        (support hint (by simp)) agree.2
      simp only [StoredWitnessExecution.executeHints, valueEq]
      apply ih _ _ (start + 1) (fun item member => support item (by simp [member]))
      exact write_agree allowed left right start _ agree

theorem instruction_agree (allowed : Nat → Prop) (left right : Array F)
    (instruction : WitnessInstruction)
    (aSupport : instruction.a.toR1CS.VarsSatisfy allowed)
    (bSupport : instruction.b.toR1CS.VarsSatisfy allowed)
    (agree : Agree allowed left right) :
    Agree allowed (StoredInstructionExecution.execute instruction left)
      (StoredInstructionExecution.execute instruction right) := by
  have aEq := instruction.a.toR1CS.eval_eq_of_agree allowed (asEnv left) (asEnv right)
    aSupport agree.2
  have bEq := instruction.b.toR1CS.eval_eq_of_agree allowed (asEnv left) (asEnv right)
    bSupport agree.2
  simp only [StoredInstructionExecution.execute, aEq, bEq]
  exact write_agree allowed left right instruction.target _ agree

theorem permutation_agree (allowed : Nat → Prop) (left right : Array F)
    (invocation : PermutationInvocation)
    (support : ∀ lane, lane < 8 →
      (invocationInputCombination invocation lane).toR1CS.VarsSatisfy allowed)
    (agree : Agree allowed left right) :
    Agree allowed (StoredPermutationExecution.execute invocation left)
      (StoredPermutationExecution.execute invocation right) := by
  have inputsEqual : StoredPermutationExecution.localInput invocation left =
      StoredPermutationExecution.localInput invocation right := by
    apply congrArg Array.ofFn
    funext column
    split_ifs with input
    · exact (invocationInputCombination invocation column.val).toR1CS.eval_eq_of_agree
        allowed (asEnv left) (asEnv right) (support _ input) agree.2
    · rfl
  have completedEqual : StoredPermutationExecution.localCompleted invocation left =
      StoredPermutationExecution.localCompleted invocation right := by
    unfold StoredPermutationExecution.localCompleted
    rw [inputsEqual]
  simp only [StoredPermutationExecution.execute, completedEqual]
  apply recipes_agree allowed left right invocation.witnessStart _ _ agree
  intro recipe member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  trivial

def CompactRowSupported (allowed : Nat → Prop) (inputColumn : Nat → Nat)
    (localStart : Nat) (row : CompactTemplateRow) : Prop :=
  (CompactRows.instantiateCombination inputColumn localStart row.a).VarsSatisfy allowed ∧
  (CompactRows.instantiateCombination inputColumn localStart row.b).VarsSatisfy allowed ∧
  (CompactRows.instantiateCombination inputColumn localStart row.c).VarsSatisfy allowed

theorem compact_step_agree (allowed : Nat → Prop) (inputColumn : Nat → Nat)
    (localStart localCount : Nat) (left right : Array F) (row : CompactTemplateRow)
    (support : CompactRowSupported allowed inputColumn localStart row)
    (agree : Agree allowed left right) :
    Option.Rel (Agree allowed)
      (StoredCompactRowExecution.step inputColumn localStart localCount left row)
      (StoredCompactRowExecution.step inputColumn localStart localCount right row) := by
  have aEq := (CompactRows.instantiateCombination inputColumn localStart row.a).eval_eq_of_agree
    allowed (asEnv left) (asEnv right) support.1 agree.2
  have bEq := (CompactRows.instantiateCombination inputColumn localStart row.b).eval_eq_of_agree
    allowed (asEnv left) (asEnv right) support.2.1 agree.2
  cases output : row.outputLocal with
  | none =>
      have cEq := (CompactRows.instantiateCombination inputColumn localStart row.c).eval_eq_of_agree
        allowed (asEnv left) (asEnv right) support.2.2 agree.2
      simp only [StoredCompactRowExecution.step, output, aEq, bEq, cEq]
      split_ifs <;> [exact .some agree; exact .none]
  | some index =>
      have updated := write_agree allowed left right (localStart + index)
        ((CompactRows.instantiateCombination inputColumn localStart row.a).eval (asEnv right) *
          (CompactRows.instantiateCombination inputColumn localStart row.b).eval (asEnv right)) agree
      have cEq := (CompactRows.instantiateCombination inputColumn localStart row.c).eval_eq_of_agree
        allowed _ _ support.2.2 updated.2
      simp only [StoredCompactRowExecution.step, output, aEq, bEq]
      by_cases inside : index < localCount
      · simp only [if_pos inside, cEq]
        split_ifs <;> [exact .some updated; exact .none]
      · simp only [if_neg inside]
        exact .none

theorem compact_run_agree (allowed : Nat → Prop) (inputColumn : Nat → Nat)
    (localStart localCount : Nat) (left right : Array F) (rows : List CompactTemplateRow)
    (support : ∀ row ∈ rows, CompactRowSupported allowed inputColumn localStart row)
    (agree : Agree allowed left right) :
    Option.Rel (Agree allowed)
      (StoredCompactRowExecution.run inputColumn localStart localCount left rows)
      (StoredCompactRowExecution.run inputColumn localStart localCount right rows) := by
  induction rows generalizing left right with
  | nil => exact .some agree
  | cons row rest ih =>
      have first := compact_step_agree allowed inputColumn localStart localCount left right row
        (support row (by simp)) agree
      cases leftStep : StoredCompactRowExecution.step inputColumn localStart localCount left row with
      | none =>
          cases rightStep : StoredCompactRowExecution.step inputColumn localStart localCount right row with
          | none => simp [StoredCompactRowExecution.run, leftStep, rightStep]
          | some result => simp [leftStep, rightStep] at first
      | some firstLeft =>
          cases rightStep : StoredCompactRowExecution.step inputColumn localStart localCount right row with
          | none => simp [leftStep, rightStep] at first
          | some firstRight =>
              have related : Agree allowed firstLeft firstRight := by
                simpa only [leftStep, rightStep, Option.rel_some_some] using first
              simpa only [StoredCompactRowExecution.run, leftStep, rightStep,
                Option.bind_some] using
                ih firstLeft firstRight (fun item member => support item (by simp [member])) related

theorem compact_agree (allowed : Nat → Prop) (inputColumn : Nat → Nat)
    (localStart : Nat) (template : CompactRowTemplate) (left right : Array F)
    (recipeSupport : template.outputRecipe.VarsSatisfy (fun input => allowed (inputColumn input)))
    (rowSupport : ∀ row ∈ template.rows,
      CompactRowSupported allowed inputColumn localStart row)
    (agree : Agree allowed left right) :
    Option.Rel (Agree allowed)
      (StoredCompactRowExecution.execute inputColumn localStart template left)
      (StoredCompactRowExecution.execute inputColumn localStart template right) := by
  have recipeEq := template.outputRecipe.eval_eq_of_agree_satisfy
    (fun input => allowed (inputColumn input)) _ _ recipeSupport
    (fun input supported => agree.2 (inputColumn input) supported)
  simp only [StoredCompactRowExecution.execute, agree.1, recipeEq]
  split_ifs
  · apply compact_run_agree allowed inputColumn localStart template.localColumnCount _ _
      template.rows rowSupport
    exact write_agree allowed left right _ _ agree
  · exact .none

end NightstreamFPrime.Export.Stage1.StoredExecutionSupport
