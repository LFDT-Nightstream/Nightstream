import Mathlib.Algebra.BigOperators.Group.List.Basic
import NightstreamFPrime.Circuit.VariableSupport

/-! Balance witness-expression sums before encoding them as nested arrays.
Values, variable support, batch order, and allocated columns are preserved.
The existing expression codec and four hint constructors remain unchanged. -/

namespace NightstreamFPrime.Export.WitnessEncoding

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

private def sumTree (terms : List Expr) : Expr :=
  if multiple : 1 < terms.length then
    .add (sumTree (terms.take (terms.length / 2))) (sumTree (terms.drop (terms.length / 2)))
  else terms.headD (.const 0)
termination_by terms.length
decreasing_by all_goals simp only [List.length_take, List.length_drop]; omega

private theorem sumTree_eval (env : Env) (terms : List Expr) :
    (sumTree terms).eval env = (terms.map (Expr.eval env)).sum := by
  induction terms using (measure List.length).wf.induction with
  | h terms ih =>
    rw [sumTree]
    split_ifs with multiple
    · rw [Expr.eval, ih (terms.take (terms.length / 2)) (by change (terms.take (terms.length / 2)).length < terms.length; simp only [List.length_take]; omega),
        ih (terms.drop (terms.length / 2)) (by change (terms.drop (terms.length / 2)).length < terms.length; simp only [List.length_drop]; omega)]
      rw [← List.sum_append, ← List.map_append, List.take_append_drop]
    · cases terms with
      | nil => rfl
      | cons first rest =>
        have empty : rest = [] := by apply List.eq_nil_of_length_eq_zero; simp only [List.length_cons] at multiple; omega
        subst rest
        simp

private theorem sumTree_support (allowed : Nat → Prop) (terms : List Expr) :
    (sumTree terms).VarsSatisfy allowed ↔ ∀ term ∈ terms, term.VarsSatisfy allowed := by
  induction terms using (measure List.length).wf.induction with
  | h terms ih =>
    rw [sumTree]
    split_ifs with multiple
    · rw [Expr.VarsSatisfy, ih (terms.take (terms.length / 2)) (by change (terms.take (terms.length / 2)).length < terms.length; simp only [List.length_take]; omega),
        ih (terms.drop (terms.length / 2)) (by change (terms.drop (terms.length / 2)).length < terms.length; simp only [List.length_drop]; omega),
        ← List.forall_mem_append, List.take_append_drop]
    · cases terms with
      | nil => simp [Expr.VarsSatisfy]
      | cons first rest =>
        have empty : rest = [] := by apply List.eq_nil_of_length_eq_zero; simp only [List.length_cons] at multiple; omega
        subst rest
        simp

private def summands : Expr → List Expr
  | .add left right => summands left ++ summands right
  | expression => [expression]

private theorem summands_eval (env : Env) (expression : Expr) :
    ((summands expression).map (Expr.eval env)).sum = expression.eval env := by
  induction expression with
  | var _ => simp [summands]
  | const _ => simp [summands]
  | mul _ _ _ _ => simp [summands]
  | add left right leftIH rightIH =>
      simp only [summands, List.map_append, List.sum_append, leftIH, rightIH, Expr.eval]

private theorem summands_support (allowed : Nat → Prop) (expression : Expr) :
    (∀ term ∈ summands expression, term.VarsSatisfy allowed) ↔ expression.VarsSatisfy allowed := by
  induction expression with
  | var _ => simp [summands]
  | const _ => simp [summands]
  | mul _ _ _ _ => simp [summands]
  | add left right leftIH rightIH =>
      simp only [summands, List.forall_mem_append, leftIH, rightIH, Expr.VarsSatisfy]

def expression : Expr → Expr
  | .var index => .var index
  | .const value => .const value
  | .mul left right => .mul (expression left) (expression right)
  | .add left right => sumTree (summands (expression left) ++ summands (expression right))

theorem expression_eval (env : Env) (value : Expr) :
    (expression value).eval env = value.eval env := by
  induction value with
  | var _ => rfl
  | const _ => rfl
  | mul left right leftIH rightIH => simp only [expression, Expr.eval, leftIH, rightIH]
  | add left right leftIH rightIH =>
      rw [expression, sumTree_eval, List.map_append, List.sum_append,
        summands_eval, summands_eval, leftIH, rightIH]
      rfl

theorem expression_support (allowed : Nat → Prop) (value : Expr) :
    (expression value).VarsSatisfy allowed ↔ value.VarsSatisfy allowed := by
  induction value with
  | var _ => rfl
  | const _ => rfl
  | mul left right leftIH rightIH => simp only [expression, Expr.VarsSatisfy, leftIH, rightIH]
  | add left right leftIH rightIH =>
      rw [expression, sumTree_support, List.forall_mem_append,
        summands_support, summands_support, leftIH, rightIH]
      rfl

def hint : Hint → Hint
  | .bit source index => .bit (expression source) index
  | .inverseOrZero source => .inverseOrZero (expression source)
  | .quotientFive source => .quotientFive (expression source)
  | .remainderFive source => .remainderFive (expression source)

theorem hint_eval (env : Env) (value : Hint) :
    (hint value).eval env = value.eval env := by
  cases value <;> simp only [hint, Hint.eval, expression_eval]

theorem hint_support (allowed : Nat → Prop) (value : Hint) :
    (hint value).source.VarsSatisfy allowed ↔ value.source.VarsSatisfy allowed := by
  cases value <;> exact expression_support allowed _

def batch (value : WitnessBatch) : WitnessBatch where
  start := value.start
  recipes := value.recipes.map expression
  hints := value.hints.map hint

theorem batch_shape (value : WitnessBatch) :
    (batch value).start = value.start ∧ (batch value).recipes.length = value.recipes.length ∧
      (batch value).hints.length = value.hints.length := by
  simp [batch]

private theorem recipes_execute (env : Env) (start : Nat) (recipes : List Expr) :
    executeRecipes env start (recipes.map expression) = executeRecipes env start recipes := by
  induction recipes generalizing env start with
  | nil => rfl
  | cons first rest ih => simp only [List.map_cons, executeRecipes, expression_eval, ih]

private theorem hints_execute (env : Env) (start : Nat) (hints : List Hint) :
    executeHints env start (hints.map hint) = executeHints env start hints := by
  induction hints generalizing env start with
  | nil => rfl
  | cons first rest ih => simp only [List.map_cons, executeHints, hint_eval, ih]

def executeBatch (env : Env) (value : WitnessBatch) : Env :=
  executeHints (executeRecipes env value.start value.recipes)
    (value.start + value.recipes.length) value.hints

theorem batch_execute (env : Env) (value : WitnessBatch) :
    executeBatch env (batch value) = executeBatch env value := by
  simp only [executeBatch, batch, List.length_map, recipes_execute, hints_execute]

theorem batches_execute (env : Env) (values : List WitnessBatch) :
    (values.map batch).foldl executeBatch env = values.foldl executeBatch env := by
  induction values generalizing env with
  | nil => rfl
  | cons first rest ih => simp only [List.map_cons, List.foldl_cons, batch_execute, ih]

end NightstreamFPrime.Export.WitnessEncoding
