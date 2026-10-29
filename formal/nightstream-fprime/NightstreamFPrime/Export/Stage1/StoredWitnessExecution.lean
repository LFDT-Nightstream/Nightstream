import NightstreamFPrime.Circuit.StraightLine

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredWitnessExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def asEnv (values : Array F) : Env :=
  fun index => values[index]?.getD 0

def write (values : Array F) (target : Nat) (value : F) : Array F :=
  values.setIfInBounds target value

theorem write_size (values : Array F) (target : Nat) (value : F) :
    (write values target value).size = values.size := by
  exact Array.size_setIfInBounds

theorem asEnv_write (values : Array F) (target : Nat) (value : F)
    (bounded : target < values.size) :
    asEnv (write values target value) = Env.set (asEnv values) target value := by
  funext index
  by_cases same : index = target
  · subst index
    simp only [asEnv, write, Env.set,
      Array.getElem?_setIfInBounds_self_of_lt bounded, Option.getD_some]
    rfl
  · simp only [asEnv, write, Env.set, ite_eq_right same,
      Array.getElem?_setIfInBounds_ne (Ne.symm same)]

def executeRecipes : Array F → Nat → List Expr → Array F
  | values, _, [] => values
  | values, start, recipe :: rest =>
      executeRecipes (write values start (recipe.eval (asEnv values))) (start + 1) rest

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
