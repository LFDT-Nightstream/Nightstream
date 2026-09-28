import NightstreamFPrime.Circuit.StraightLine

/-!
Array storage for the existing sequential recipe and hint interpreters.
Reads outside the array return zero; writes outside it leave storage intact.
The refinement theorems require every write target to be in bounds. They
establish execution equality, not constraint satisfaction or hint authority.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredWitnessExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- Total environment view of the stored field values. -/
def asEnv (values : Array F) : Env :=
  fun index => values[index]?.getD 0

/-- Update an existing cell without changing the storage size. -/
def write (values : Array F) (target : Nat) (value : F) : Array F :=
  values.setIfInBounds target value

theorem write_size (values : Array F) (target : Nat) (value : F) :
    (write values target value).size = values.size := by
  exact Array.size_setIfInBounds

/-- Every read of an in-bounds stored update equals the functional update,
including reads outside the array. -/
theorem asEnv_write (values : Array F) (target : Nat) (value : F)
    (bounded : target < values.size) :
    asEnv (write values target value) = Env.set (asEnv values) target value := by
  funext index
  by_cases same : index = target
  · subst index
    simp only [asEnv, write, Env.set,
      Array.getElem?_setIfInBounds_self_of_lt bounded, Option.getD_some]
    rfl
  · simp only [asEnv, write, Env.set, if_neg same,
      Array.getElem?_setIfInBounds_ne (Ne.symm same)]

/-- Evaluate each recipe on the current array environment, write its result,
then continue at the next target. Expression arithmetic has its existing owner. -/
def executeRecipes : Array F → Nat → List Expr → Array F
  | values, _, [] => values
  | values, start, recipe :: rest =>
      executeRecipes (write values start (recipe.eval (asEnv values))) (start + 1) rest

/-- Hint writes follow the same sequential order as Circuit.executeHints.
Hint evaluation and the later constraints on its results remain unchanged. -/
def executeHints : Array F → Nat → List Hint → Array F
  | values, _, [] => values
  | values, start, hint :: rest =>
      executeHints (write values start (hint.eval (asEnv values))) (start + 1) rest

theorem executeRecipes_size (values : Array F) (start : Nat) (recipes : List Expr) :
    (executeRecipes values start recipes).size = values.size := by
  induction recipes generalizing values start with
  | nil => rfl
  | cons recipe rest inductionHypothesis =>
      rw [executeRecipes, inductionHypothesis, write_size]

theorem executeHints_size (values : Array F) (start : Nat) (hints : List Hint) :
    (executeHints values start hints).size = values.size := by
  induction hints generalizing values start with
  | nil => rfl
  | cons hint rest inductionHypothesis =>
      rw [executeHints, inductionHypothesis, write_size]

/-- Arbitrary recipes produce the exact functional environment when their
whole write interval fits the array. No expression-read restriction is needed. -/
theorem executeRecipes_eq (values : Array F) (start : Nat) (recipes : List Expr)
    (fits : start + recipes.length ≤ values.size) :
    asEnv (executeRecipes values start recipes) =
      NightstreamFPrime.Circuit.executeRecipes (asEnv values) start recipes := by
  induction recipes generalizing values start with
  | nil => rfl
  | cons recipe rest inductionHypothesis =>
      have bounded : start < values.size := by
        simp only [List.length_cons] at fits
        omega
      have remaining : start + 1 + rest.length ≤
          (write values start (recipe.eval (asEnv values))).size := by
        rw [write_size]
        simp only [List.length_cons] at fits
        omega
      rw [executeRecipes, NightstreamFPrime.Circuit.executeRecipes,
        inductionHypothesis (write values start (recipe.eval (asEnv values))) (start + 1) remaining,
        asEnv_write values start (recipe.eval (asEnv values)) bounded]

/-- Arbitrary hints produce the exact functional environment under the same
write-range bound. The theorem gives no authority to unconstrained hints. -/
theorem executeHints_eq (values : Array F) (start : Nat) (hints : List Hint)
    (fits : start + hints.length ≤ values.size) :
    asEnv (executeHints values start hints) =
      NightstreamFPrime.Circuit.executeHints (asEnv values) start hints := by
  induction hints generalizing values start with
  | nil => rfl
  | cons hint rest inductionHypothesis =>
      have bounded : start < values.size := by
        simp only [List.length_cons] at fits
        omega
      have remaining : start + 1 + rest.length ≤
          (write values start (hint.eval (asEnv values))).size := by
        rw [write_size]
        simp only [List.length_cons] at fits
        omega
      rw [executeHints, NightstreamFPrime.Circuit.executeHints,
        inductionHypothesis (write values start (hint.eval (asEnv values))) (start + 1) remaining,
        asEnv_write values start (hint.eval (asEnv values)) bounded]

end NightstreamFPrime.Export.Stage1.StoredWitnessExecution
