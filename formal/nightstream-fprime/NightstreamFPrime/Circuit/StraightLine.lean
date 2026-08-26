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

theorem VarsBelow.add (left right : Expr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (left + right).VarsBelow bound :=
  ⟨leftBelow, rightBelow⟩

theorem VarsBelow.mul (left right : Expr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (left * right).VarsBelow bound :=
  ⟨leftBelow, rightBelow⟩

theorem VarsBelow.neg (expression : Expr) (bound : Nat)
    (below : expression.VarsBelow bound) :
    (-expression).VarsBelow bound :=
  ⟨trivial, below⟩

theorem VarsBelow.sub (left right : Expr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (left - right).VarsBelow bound :=
  ⟨leftBelow, ⟨trivial, rightBelow⟩⟩

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

/-- Every authoritative recipe row reads only the complete batch prefix. -/
theorem recipeConstraints_varsBelow_of_causal (start : Nat)
    (recipes : List Expr) (causal : RecipesCausal start recipes) :
    ∀ expression ∈ recipeConstraints start recipes,
      expression.VarsBelow (start + recipes.length) := by
  induction recipes generalizing start with
  | nil =>
      intro expression member
      simp [recipeConstraints] at member
  | cons recipe rest inductionHypothesis =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · apply Expr.VarsBelow.sub
        · unfold Expr.VarsBelow
          simp only [List.length_cons]
          omega
        · exact Expr.VarsBelow.mono recipe causal.1 (by
            simp only [List.length_cons]
            omega)
      · have below := inductionHypothesis (start := start + 1)
          causal.2 expression member
        convert below using 1 <;> simp only [List.length_cons] <;> omega

/-- Satisfaction is stable when every referenced variable is unchanged. -/
theorem constraintsHold_of_agree_below
    (before after : Env) (constraints : List Expr) (bound : Nat)
    (scope : ∀ expression ∈ constraints, expression.VarsBelow bound)
    (agrees : ∀ index, index < bound → after index = before index)
    (holdsBefore : ConstraintsHold before constraints) :
    ConstraintsHold after constraints := by
  intro expression member
  rw [expression.eval_eq_of_agree_below bound after before
    (scope expression member) agrees]
  exact holdsBefore expression member

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

/-- Recipe constraints split at the exact variable offset allocated by the
first recipe list. -/
theorem recipeConstraints_append (start : Nat)
    (first second : List Expr) :
    recipeConstraints start (first ++ second) =
      recipeConstraints start first ++
        recipeConstraints (start + first.length) second := by
  induction first generalizing start with
  | nil => simp [recipeConstraints]
  | cons recipe rest inductionHypothesis =>
      simp only [List.cons_append, recipeConstraints, List.length_cons,
        List.cons.injEq, true_and]
      rw [inductionHypothesis]
      congr 2
      omega

/-- Satisfaction of an appended constraint list is exactly satisfaction of
both parts. -/
theorem constraintsHold_append (env : Env) (first second : List Expr) :
    ConstraintsHold env (first ++ second) ↔
      ConstraintsHold env first ∧ ConstraintsHold env second := by
  constructor
  · intro holds
    exact ⟨
      fun expression member =>
        holds expression (List.mem_append_left second member),
      fun expression member =>
        holds expression (List.mem_append_right first member)⟩
  · rintro ⟨firstHolds, secondHolds⟩ expression member
    rcases List.mem_append.mp member with member | member
    · exact firstHolds expression member
    · exact secondHolds expression member

/-- Read the exact equality carried by one indexed straight-line recipe row. -/
theorem recipeConstraints_value (env : Env) (start : Nat)
    (recipes : List Expr) (rows : ConstraintsHold env
      (recipeConstraints start recipes))
    (index : Nat) (bounded : index < recipes.length) :
    env (start + index) =
      (recipes.get ⟨index, bounded⟩).eval env := by
  induction recipes generalizing start index with
  | nil => simp at bounded
  | cons recipe rest inductionHypothesis =>
      cases index with
      | zero =>
          have row := rows (Expr.var start - recipe) (by
            simp [recipeConstraints])
          have zero : env start - recipe.eval env = 0 := by
            simpa [Expr.eval_sub] using row
          exact sub_eq_zero.mp zero
      | succ index =>
          have tailBound : index < rest.length := by
            simpa using bounded
          have tailRows : ConstraintsHold env
              (recipeConstraints (start + 1) rest) := by
            intro expression member
            exact rows expression (by simp [recipeConstraints, member])
          have value := inductionHypothesis (start + 1) tailRows index tailBound
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using value

/-- Indexed recipe equalities satisfy the complete straight-line row list. -/
theorem recipeConstraints_hold_of_values (env : Env) (start : Nat)
    (recipes : List Expr)
    (values : ∀ index (bounded : index < recipes.length),
      env (start + index) = (recipes.get ⟨index, bounded⟩).eval env) :
    ConstraintsHold env (recipeConstraints start recipes) := by
  induction recipes generalizing start with
  | nil =>
      intro expression member
      simp [recipeConstraints] at member
  | cons recipe rest inductionHypothesis =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | tailMember
      · have head := values 0 (by simp)
        have headEq : env start = recipe.eval env := by
          simpa using head
        have zero : env start - recipe.eval env = 0 := by
          rw [headEq]
          exact sub_self _
        simpa [Expr.eval_sub] using zero
      · apply inductionHypothesis (start + 1)
        · intro index bounded
          have value := values (index + 1) (by simpa using bounded)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using value
        · exact tailMember

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

/-! ## Non-authoritative hint execution -/

/-- Every hint in one batch reads only caller-owned values below its start. -/
def HintsReadBelow (start : Nat) (hints : List Hint) : Prop :=
  ∀ hint ∈ hints, hint.source.VarsBelow start

theorem Hint.eval_eq_of_agree_below (hint : Hint) (bound : Nat)
    (left right : Env) (below : hint.source.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    hint.eval left = hint.eval right := by
  cases hint with
  | bit source index =>
      simp only [Hint.eval]
      rw [source.eval_eq_of_agree_below bound left right below agrees]
  | inverseOrZero source =>
      simp only [Hint.eval]
      rw [source.eval_eq_of_agree_below bound left right below agrees]
  | quotientFive source =>
      simp only [Hint.eval]
      rw [source.eval_eq_of_agree_below bound left right below agrees]
  | remainderFive source =>
      simp only [Hint.eval]
      rw [source.eval_eq_of_agree_below bound left right below agrees]

/-- Execute a hint list in order. Hints remain non-authoritative until later
rows constrain their output variables. -/
def executeHints : Env → Nat → List Hint → Env
  | env, _, [] => env
  | env, start, hint :: rest =>
      executeHints (Env.set env start (hint.eval env)) (start + 1) rest

theorem executeHints_agrees_below
    (env : Env) (start : Nat) (hints : List Hint) :
    ∀ index, index < start → executeHints env start hints index = env index := by
  induction hints generalizing env start with
  | nil =>
      intro index _
      rfl
  | cons hint rest inductionHypothesis =>
      intro index indexLt
      rw [executeHints]
      calc
        executeHints (Env.set env start (hint.eval env)) (start + 1) rest index =
            Env.set env start (hint.eval env) index :=
          inductionHypothesis _ _ index (by omega)
        _ = env index := Env.set_of_ne env start index _ (by omega)

theorem executeHints_agrees_above
    (env : Env) (start : Nat) (hints : List Hint) :
    ∀ index, start + hints.length ≤ index →
      executeHints env start hints index = env index := by
  induction hints generalizing env start with
  | nil =>
      intro index _
      rfl
  | cons hint rest inductionHypothesis =>
      intro index indexGe
      rw [executeHints]
      calc
        executeHints (Env.set env start (hint.eval env)) (start + 1) rest index =
            Env.set env start (hint.eval env) index :=
          inductionHypothesis _ _ index (by
            simp only [List.length_cons] at indexGe
            omega)
        _ = env index := Env.set_of_ne env start index _ (by
          simp only [List.length_cons] at indexGe
          omega)

theorem executeHints_agreesOutside
    (env : Env) (start : Nat) (hints : List Hint) :
    AgreesOutside env (executeHints env start hints) start hints.length := by
  intro index outside
  rcases outside with below | above
  · exact executeHints_agrees_below env start hints index below
  · exact executeHints_agrees_above env start hints index above

/-- An external-read hint has its exact executable value at its assigned
slot after the whole batch completes. -/
theorem executeHints_value_of_readBelow
    (env : Env) (start : Nat) (hints : List Hint)
    (readBelow : HintsReadBelow start hints)
    (position : Nat) (positionLt : position < hints.length) :
    executeHints env start hints (start + position) =
      (hints.get ⟨position, positionLt⟩).eval env := by
  induction hints generalizing env start position with
  | nil =>
      simp at positionLt
  | cons hint rest inductionHypothesis =>
      cases position with
      | zero =>
          let assigned := Env.set env start (hint.eval env)
          calc
            executeHints env start (hint :: rest) start =
                executeHints assigned (start + 1) rest start := by
              rfl
            _ = assigned start :=
              executeHints_agrees_below assigned (start + 1) rest start
                (by omega)
            _ = hint.eval env := Env.set_self env start _
            _ = ((hint :: rest).get ⟨0, positionLt⟩).eval env := by rfl
      | succ position =>
          let assigned := Env.set env start (hint.eval env)
          have tailPositionLt : position < rest.length := by
            simpa using positionLt
          have tailReadBelow : HintsReadBelow (start + 1) rest := by
            intro tailHint member
            exact Expr.VarsBelow.mono tailHint.source
              (readBelow tailHint (by simp [member])) (by omega)
          have selectedBelow :
              (rest.get ⟨position, tailPositionLt⟩).source.VarsBelow start :=
            readBelow _ (by simp)
          calc
            executeHints env start (hint :: rest) (start + (position + 1)) =
                executeHints assigned (start + 1) rest
                  ((start + 1) + position) := by
              rw [executeHints]
              apply congrArg (fun index =>
                executeHints assigned (start + 1) rest index)
              omega
            _ = (rest.get ⟨position, tailPositionLt⟩).eval assigned :=
              inductionHypothesis assigned (start + 1) tailReadBelow position
                tailPositionLt
            _ = (rest.get ⟨position, tailPositionLt⟩).eval env := by
              apply Hint.eval_eq_of_agree_below _ start assigned env selectedBelow
              intro index indexLt
              exact Env.set_of_ne env start index _ (by omega)
            _ = ((hint :: rest).get ⟨position + 1, positionLt⟩).eval env := by
              rfl

end NightstreamFPrime.Circuit
