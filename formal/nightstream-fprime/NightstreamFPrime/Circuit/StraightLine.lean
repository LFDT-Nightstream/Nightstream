import NightstreamFPrime.Circuit.Basic

/-!
Owns execution and correctness of the canonical straight-line witness IR.
Recipes may read external variables and earlier recipe results only. The
interpreter is not circuit semantics: emitted rows check every computed value.
-/

namespace NightstreamFPrime.Circuit

open NightstreamFPrime.Spec

namespace Expr

/-- Every variable read by an expression has an index below `bound`. -/
def VarsBelow (bound : Nat) : Expr → Prop
  | .var index => index < bound
  | .const _ => True
  | .add left right => left.VarsBelow bound ∧ right.VarsBelow bound
  | .mul left right => left.VarsBelow bound ∧ right.VarsBelow bound

theorem eval_eq_of_agree_below (expression : Expr) (bound : Nat) (left right : Env)
    (hvars : expression.VarsBelow bound)
    (hagrees : ∀ index, index < bound → left index = right index) :
    expression.eval left = expression.eval right := by
  induction expression with
  | var index =>
      exact hagrees index hvars
  | const value =>
      rfl
  | add a b ha hb =>
      exact congrArg₂ (· + ·)
        (ha hvars.1) (hb hvars.2)
  | mul a b ha hb =>
      exact congrArg₂ (· * ·)
        (ha hvars.1) (hb hvars.2)

end Expr

namespace Env

/-- Functional update used by the exported witness interpreter. -/
def set (env : Env) (index : Nat) (value : F) : Env :=
  fun current => if current = index then value else env current

@[simp] theorem set_self (env : Env) (index : Nat) (value : F) :
    set env index value index = value := by
  simp [set]

theorem set_of_ne (env : Env) (index current : Nat) (value : F)
    (hne : current ≠ index) : set env index value current = env current := by
  simp [set, hne]

end Env

/-- Causality for one witness batch: recipe `i` reads only external values or
results produced before `start + i`. -/
def RecipesCausal : Nat → List Expr → Prop
  | _, [] => True
  | start, recipe :: rest =>
      recipe.VarsBelow start ∧ RecipesCausal (start + 1) rest

theorem Expr.VarsBelow.mono (expression : Expr) {lower upper : Nat}
    (hvars : expression.VarsBelow lower) (hle : lower ≤ upper) :
    expression.VarsBelow upper := by
  induction expression with
  | var index =>
      simp only [VarsBelow] at hvars ⊢
      omega
  | const value =>
      trivial
  | add left right hleft hright =>
      exact ⟨hleft hvars.1, hright hvars.2⟩
  | mul left right hleft hright =>
      exact ⟨hleft hvars.1, hright hvars.2⟩

theorem recipesCausal_of_all_below (start : Nat) (recipes : List Expr)
    (hbelow : ∀ expression ∈ recipes, expression.VarsBelow start) :
    RecipesCausal start recipes := by
  induction recipes generalizing start with
  | nil =>
      trivial
  | cons recipe rest ih =>
      constructor
      · exact hbelow recipe (by simp)
      · apply ih (start := start + 1)
        intro expression hmem
        exact Expr.VarsBelow.mono expression
          (hbelow expression (by simp [hmem])) (by omega)

/-- Appending a batch whose recipes read only the completed prefix preserves
causality. The proof is structural in the prefix and suffix lists. -/
theorem recipesCausal_append (start : Nat) (existing added : List Expr)
    (hexisting : RecipesCausal start existing)
    (hadded : ∀ expression ∈ added,
      expression.VarsBelow (start + existing.length)) :
    RecipesCausal start (existing ++ added) := by
  induction existing generalizing start with
  | nil =>
      simpa using recipesCausal_of_all_below start added hadded
  | cons recipe rest ih =>
      constructor
      · exact hexisting.1
      · apply ih (start := start + 1) hexisting.2
        intro expression hmem
        have h := hadded expression hmem
        convert h using 1 <;> simp only [List.length_cons] <;> omega

/-- Execute the canonical witness recipes in order. -/
def executeRecipes : Env → Nat → List Expr → Env
  | env, _, [] => env
  | env, start, recipe :: rest =>
      executeRecipes (Env.set env start (recipe.eval env)) (start + 1) rest

theorem executeRecipes_agrees_below (env : Env) (start : Nat) (recipes : List Expr) :
    ∀ index, index < start → executeRecipes env start recipes index = env index := by
  induction recipes generalizing env start with
  | nil =>
      intro index _
      rfl
  | cons recipe rest ih =>
      intro index hindex
      rw [executeRecipes]
      calc
        executeRecipes (Env.set env start (recipe.eval env)) (start + 1) rest index =
            Env.set env start (recipe.eval env) index :=
          ih _ _ index (by omega)
        _ = env index := Env.set_of_ne env start index _ (by omega)

theorem executeRecipes_agrees_above (env : Env) (start : Nat) (recipes : List Expr) :
    ∀ index, start + recipes.length ≤ index →
      executeRecipes env start recipes index = env index := by
  induction recipes generalizing env start with
  | nil =>
      intro index _
      rfl
  | cons recipe rest ih =>
      intro index hindex
      rw [executeRecipes]
      calc
        executeRecipes (Env.set env start (recipe.eval env)) (start + 1) rest index =
            Env.set env start (recipe.eval env) index :=
          ih _ _ index (by
            simp only [List.length_cons] at hindex
            omega)
        _ = env index := Env.set_of_ne env start index _ (by
          simp only [List.length_cons] at hindex
          omega)

theorem executeRecipes_agreesOutside (env : Env) (start : Nat) (recipes : List Expr) :
    AgreesOutside env (executeRecipes env start recipes) start recipes.length := by
  intro index houtside
  rcases houtside with hbelow | habove
  · exact executeRecipes_agrees_below env start recipes index hbelow
  · exact executeRecipes_agrees_above env start recipes index habove

/-- Honest execution of a causal batch satisfies every authoritative recipe
row. The proof is structural in the recipe list. -/
theorem executeRecipes_holds_recipeConstraints (env : Env) (start : Nat)
    (recipes : List Expr) (hcausal : RecipesCausal start recipes) :
    ConstraintsHold (executeRecipes env start recipes)
      (recipeConstraints start recipes) := by
  induction recipes generalizing env start with
  | nil =>
      intro expression hmem
      simp [recipeConstraints] at hmem
  | cons recipe rest ih =>
      let assigned := Env.set env start (recipe.eval env)
      let completed := executeRecipes assigned (start + 1) rest
      have hrecipe : recipe.VarsBelow start := hcausal.1
      have hrest : RecipesCausal (start + 1) rest := hcausal.2
      have hbelow : ∀ index, index < start → completed index = env index := by
        intro index hindex
        calc
          completed index = assigned index :=
            executeRecipes_agrees_below assigned (start + 1) rest index (by omega)
          _ = env index := Env.set_of_ne env start index _ (by omega)
      have hvalue : completed start = recipe.eval env := by
        calc
          completed start = assigned start :=
            executeRecipes_agrees_below assigned (start + 1) rest start (by omega)
          _ = recipe.eval env := Env.set_self env start _
      have heval : recipe.eval completed = recipe.eval env :=
        recipe.eval_eq_of_agree_below start completed env hrecipe hbelow
      intro expression hmem
      simp only [recipeConstraints, List.mem_cons] at hmem
      rcases hmem with rfl | hmem
      · rw [executeRecipes]
        change (Expr.var start - recipe).eval completed = 0
        simp only [Expr.eval_sub, Expr.eval_var, hvalue, heval, sub_self]
      · change expression.eval completed = 0
        exact ih assigned (start + 1) hrest expression hmem

end NightstreamFPrime.Circuit
