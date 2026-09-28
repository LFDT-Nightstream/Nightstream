import NightstreamFPrime.Export.Stage1.StoredExecutionSupport
import NightstreamFPrime.Export.Stage1.StoredCompactCompletion

/-!
Stored construction of a compact template's required output only. The same
write bounds remain checked. Equivalence with full execution is proved for
canonical expression templates whose discarded cells are not observed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredCompactOutput

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open StoredWitnessExecution (asEnv write)
open StoredExecutionSupport (Agree)

/-- Evaluate the existing output recipe on the input snapshot. No local row
is evaluated and no scratch cell is written. Bounds match the full executor. -/
def execute (inputColumn : Nat → Nat) (localStart : Nat)
    (template : CompactRowTemplate) (values : Array F) : Option (Array F) :=
  if inputColumn template.outputInput < values.size ∧
      localStart + template.localColumnCount ≤ values.size then
    some (write values (inputColumn template.outputInput)
      (template.outputRecipe.eval (fun input => asEnv values (inputColumn input))))
  else none

private theorem full_output_agree (allowed : Nat → Prop)
    (inputColumn : Nat → Nat) (localStart : Nat) (template : CompactRowTemplate)
    (left right after : Array F)
    (recipeSupport : template.outputRecipe.VarsSatisfy (fun input => allowed (inputColumn input)))
    (ignored : ∀ column, allowed column →
      column < localStart ∨ localStart + template.localColumnCount ≤ column)
    (agree : Agree allowed left right)
    (success : StoredCompactRowExecution.execute inputColumn localStart template left = some after) :
    Agree allowed after
      (write right (inputColumn template.outputInput)
        (template.outputRecipe.eval (fun input => asEnv right (inputColumn input)))) := by
  have size := StoredCompactRowExecution.execute_size inputColumn localStart template left after success
  have retained := StoredCompactRowExecution.execute_agreesOutside
    inputColumn localStart template left after success
  have valueEq := template.outputRecipe.eval_eq_of_agree_satisfy
    (fun input => allowed (inputColumn input)) _ _ recipeSupport
    (fun input supported => agree.2 (inputColumn input) supported)
  have written := StoredExecutionSupport.write_agree allowed left right
    (inputColumn template.outputInput)
    (template.outputRecipe.eval (fun input => asEnv right (inputColumn input))) agree
  refine ⟨?_, ?_⟩
  · rw [StoredWitnessExecution.write_size]
    exact size.trans agree.1
  · intro column supported
    rw [retained column (ignored column supported), valueEq]
    exact written.2 column supported

/-- Full and output-only execution have the same success/rejection and
retained values. Reconstruction of the discarded expression rows is total;
the caller supplies only structural read/write geometry. -/
theorem compactTemplate_agree (allowed : Nat → Prop)
    (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
    (recipe : Expr) (left right : Array F)
    (localBound : inputCount ≤ localStart)
    (inputsOutside : ∀ input, input < inputCount →
      inputColumn input < localStart ∨ localStart +
        R1CS.mulCount (Expr.var outputInput - recipe) ≤ inputColumn input)
    (outputBelow : outputInput < inputCount)
    (recipeScope : recipe.VarsBelow outputInput)
    (outputDistinct : ∀ input, input < outputInput →
      inputColumn input ≠ inputColumn outputInput)
    (recipeSupport : recipe.VarsSatisfy (fun input => allowed (inputColumn input)))
    (ignored : ∀ column, allowed column →
      column < localStart ∨ localStart +
        R1CS.mulCount (Expr.var outputInput - recipe) ≤ column)
    (agree : Agree allowed left right) :
    Option.Rel (Agree allowed)
      (StoredCompactRowExecution.execute inputColumn localStart
        (CompactRows.compactTemplate inputCount outputInput recipe) left)
      (execute inputColumn localStart
        (CompactRows.compactTemplate inputCount outputInput recipe) right) := by
  by_cases fits : inputColumn outputInput < left.size ∧ localStart +
      R1CS.mulCount (Expr.var outputInput - recipe) ≤ left.size
  · have rightFits : inputColumn outputInput < right.size ∧ localStart +
        R1CS.mulCount (Expr.var outputInput - recipe) ≤ right.size := by
      simpa only [agree.1] using fits
    have total := StoredCompactCompletion.execute_compactTemplate inputCount outputInput
      localStart inputColumn recipe left fits.1 fits.2 localBound inputsOutside
      outputBelow recipeScope outputDistinct
    cases result : StoredCompactRowExecution.execute inputColumn localStart
        (CompactRows.compactTemplate inputCount outputInput recipe) left with
    | none =>
        simp only [result, Option.map_none] at total
        contradiction
    | some after =>
        have retained := full_output_agree allowed inputColumn localStart
          (CompactRows.compactTemplate inputCount outputInput recipe) left right after
          recipeSupport ignored agree result
        simpa only [execute, CompactRows.compactTemplate, if_pos rightFits,
          Option.rel_some_some] using retained
  · have rightFits : ¬ (inputColumn outputInput < right.size ∧ localStart +
        R1CS.mulCount (Expr.var outputInput - recipe) ≤ right.size) := by
      simpa only [agree.1] using fits
    simp only [StoredCompactRowExecution.execute, execute, CompactRows.compactTemplate,
      if_neg fits, if_neg rightFits]
    exact .none

end NightstreamFPrime.Export.Stage1.StoredCompactOutput
