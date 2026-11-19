import NightstreamFPrime.Circuit.VariableSupport

/-! Execution of a witness program (`SECURITY_MODEL.md`, trust gap 5). `Witness.execute` runs the
batches in order. `Witness.Computed` states its result without the order: each recipe row holds
and each hint column has the value of its hint. An ordered program computes its own result
(`Witness.execute_computed`), so a gadget proves its rows from `Witness.Computed` in one
environment, and a parent has the rows of each child in that environment
(`Witness.computed_append`), and its completeness follows from its order and its rows
(`Witness.complete`). A native fill that gives the values of its batches off the dead
columns keeps two executions equal off those columns (`Witness.agreeOff_fill`). -/

namespace NightstreamFPrime.Circuit

open NightstreamFPrime.Spec

def HintsCausal : Nat → List Hint → Prop
  | _, [] => True
  | start, hint :: rest => hint.source.VarsBelow start ∧ HintsCausal (start + 1) rest

def HintsComputed (env : Env) : Nat → List Hint → Prop
  | _, [] => True
  | start, hint :: rest => env start = hint.eval env ∧ HintsComputed env (start + 1) rest

theorem hintsCausal_of_readBelow (start : Nat) (hints : List Hint)
    (below : HintsReadBelow start hints) : HintsCausal start hints := by
  induction hints generalizing start with
  | nil => trivial
  | cons hint rest ih =>
      refine ⟨below hint (by simp), ih (start + 1) ?_⟩
      intro later member
      exact Expr.VarsBelow.mono _ (below later (by simp [member])) (by omega)

theorem hintsComputed_value {env : Env} {start : Nat} {hints : List Hint}
    (computed : HintsComputed env start hints) (index : Nat) (bounded : index < hints.length) :
    env (start + index) = (hints.get ⟨index, bounded⟩).eval env := by
  induction hints generalizing start index with
  | nil => simp at bounded
  | cons hint rest ih =>
      cases index with
      | zero => exact computed.1
      | succ index =>
          rw [show start + (index + 1) = start + 1 + index by omega]
          exact ih computed.2 index (by simpa using bounded)

theorem hintsComputed_of_agree {before after : Env} {start : Nat} {hints : List Hint}
    (causal : HintsCausal start hints)
    (agrees : ∀ index, index < start + hints.length → after index = before index)
    (computed : HintsComputed before start hints) : HintsComputed after start hints := by
  induction hints generalizing start with
  | nil => trivial
  | cons hint rest ih =>
      have shifted : ∀ index, index < start + 1 + rest.length → after index = before index :=
        fun index below => agrees index (by simp only [List.length_cons]; omega)
      refine ⟨?_, ih causal.2 shifted computed.2⟩
      rw [shifted start (by omega), computed.1]
      exact hint.eval_eq_of_agree_below start before after causal.1
        fun index below => (shifted index (by omega)).symm

theorem executeHints_computed (env : Env) (start : Nat) (hints : List Hint)
    (causal : HintsCausal start hints) :
    HintsComputed (executeHints env start hints) start hints := by
  induction hints generalizing env start with
  | nil => trivial
  | cons hint rest ih =>
      refine ⟨?_, ih (Env.set env start (hint.eval env)) (start + 1) causal.2⟩
      show executeHints (Env.set env start (hint.eval env)) (start + 1) rest start =
        hint.eval (executeHints (Env.set env start (hint.eval env)) (start + 1) rest)
      rw [executeHints_agrees_below _ (start + 1) rest start (by omega), Env.set_self]
      exact (hint.eval_eq_of_agree_below start _ env causal.1 fun index below =>
        (executeHints_agrees_below _ (start + 1) rest index (by omega)).trans
          (Env.set_of_ne env start index _ (by omega))).symm

theorem Hint.eval_eq_of_agree_satisfy (hint : Hint) (allowed : Nat → Prop) (left right : Env)
    (support : hint.source.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    hint.eval left = hint.eval right := by
  cases hint <;>
    (have same := Expr.eval_eq_of_agree_satisfy _ allowed left right support agrees
     simp only [Hint.source] at same
     simp only [Hint.eval, same])

namespace WitnessBatch

def hintStart (batch : WitnessBatch) : Nat := batch.start + batch.recipes.length

def finish (batch : WitnessBatch) : Nat := batch.start + batch.outputLength

/-- The order of `Export.BoundedDepth.StepPackage.batchEntries`: the recipes, then the hints. -/
def execute (env : Env) (batch : WitnessBatch) : Env :=
  executeHints (executeRecipes env batch.start batch.recipes) batch.hintStart batch.hints

def Causal (batch : WitnessBatch) : Prop :=
  RecipesCausal batch.start batch.recipes ∧ HintsCausal batch.hintStart batch.hints

def Computed (env : Env) (batch : WitnessBatch) : Prop :=
  ConstraintsHold env (recipeConstraints batch.start batch.recipes) ∧
    HintsComputed env batch.hintStart batch.hints

def Avoids (dead : Nat → Prop) (batch : WitnessBatch) : Prop :=
  (∀ recipe ∈ batch.recipes, recipe.VarsSatisfy fun column => ¬dead column) ∧
    ∀ hint ∈ batch.hints, hint.source.VarsSatisfy fun column => ¬dead column

theorem finish_eq (batch : WitnessBatch) :
    batch.finish = batch.hintStart + batch.hints.length := by
  simp [finish, hintStart, outputLength, Nat.add_assoc]

theorem execute_agrees_below (env : Env) (batch : WitnessBatch) (index : Nat)
    (below : index < batch.start) : batch.execute env index = env index := by
  unfold execute
  rw [executeHints_agrees_below _ _ _ index (by unfold hintStart; omega),
    executeRecipes_agrees_below _ _ _ index below]

theorem execute_agrees_above (env : Env) (batch : WitnessBatch) (index : Nat)
    (above : batch.finish ≤ index) : batch.execute env index = env index := by
  unfold execute
  rw [executeHints_agrees_above _ _ _ index (by rw [← finish_eq]; exact above),
    executeRecipes_agrees_above _ _ _ index (by rw [finish_eq] at above; unfold hintStart at above; omega)]

theorem computed_of_agree {before after : Env} {batch : WitnessBatch} (causal : batch.Causal)
    (agrees : ∀ index, index < batch.finish → after index = before index)
    (computed : batch.Computed before) : batch.Computed after :=
  ⟨constraintsHold_of_agree_below before after _ batch.hintStart
      (recipeConstraints_varsBelow_of_causal batch.start batch.recipes causal.1)
      (fun index below => agrees index (by rw [finish_eq]; omega)) computed.1,
    hintsComputed_of_agree causal.2
      (fun index below => agrees index (by rw [finish_eq]; exact below)) computed.2⟩

theorem execute_computed (env : Env) (batch : WitnessBatch) (causal : batch.Causal) :
    batch.Computed (batch.execute env) :=
  ⟨constraintsHold_of_agree_below (executeRecipes env batch.start batch.recipes) _ _
      batch.hintStart
      (recipeConstraints_varsBelow_of_causal batch.start batch.recipes causal.1)
      (executeHints_agrees_below _ batch.hintStart batch.hints)
      (executeRecipes_holds_recipeConstraints env batch.start batch.recipes causal.1),
    executeHints_computed _ batch.hintStart batch.hints causal.2⟩

end WitnessBatch

def AgreeOff (dead : Nat → Prop) (left right : Env) : Prop :=
  ∀ column, ¬dead column → left column = right column

theorem AgreeOff.set {dead : Nat → Prop} {left right : Env} (agree : AgreeOff dead left right)
    (index : Nat) {leftValue rightValue : F} (same : leftValue = rightValue) :
    AgreeOff dead (Env.set left index leftValue) (Env.set right index rightValue) := by
  intro column live
  by_cases here : column = index
  · subst here
    rw [Env.set_self, Env.set_self, same]
  · rw [Env.set_of_ne _ _ _ _ here, Env.set_of_ne _ _ _ _ here]
    exact agree column live

theorem executeRecipes_agreeOff {dead : Nat → Prop} (start : Nat) (recipes : List Expr)
    (avoids : ∀ recipe ∈ recipes, recipe.VarsSatisfy fun column => ¬dead column)
    {left right : Env} (agree : AgreeOff dead left right) :
    AgreeOff dead (executeRecipes left start recipes) (executeRecipes right start recipes) := by
  induction recipes generalizing start left right with
  | nil => exact agree
  | cons recipe rest ih =>
      exact ih (start + 1) (fun later member => avoids later (by simp [member]))
        (agree.set start (recipe.eval_eq_of_agree_satisfy _ left right (avoids recipe (by simp))
          agree))

theorem executeHints_agreeOff {dead : Nat → Prop} (start : Nat) (hints : List Hint)
    (avoids : ∀ hint ∈ hints, hint.source.VarsSatisfy fun column => ¬dead column)
    {left right : Env} (agree : AgreeOff dead left right) :
    AgreeOff dead (executeHints left start hints) (executeHints right start hints) := by
  induction hints generalizing start left right with
  | nil => exact agree
  | cons hint rest ih =>
      exact ih (start + 1) (fun later member => avoids later (by simp [member]))
        (agree.set start (hint.eval_eq_of_agree_satisfy _ left right (avoids hint (by simp))
          agree))

theorem WitnessBatch.execute_agreeOff {dead : Nat → Prop} {batch : WitnessBatch}
    (avoids : batch.Avoids dead) {left right : Env} (agree : AgreeOff dead left right) :
    AgreeOff dead (batch.execute left) (batch.execute right) :=
  executeHints_agreeOff _ _ avoids.2 (executeRecipes_agreeOff _ _ avoids.1 agree)

def Op.RecipesOnly : Op → Prop
  | .witness _ | .mark _ => True
  | .assertZero _ | .subcircuit _ => False

theorem executeRecipes_append (env : Env) (start : Nat) (first second : List Expr) :
    executeRecipes env start (first ++ second) =
      executeRecipes (executeRecipes env start first) (start + first.length) second := by
  induction first generalizing env start with
  | nil => rfl
  | cons recipe rest ih =>
      rw [List.cons_append, executeRecipes, ih, List.length_cons, executeRecipes,
        Nat.add_right_comm, Nat.add_assoc]

theorem recipeConstraints_split : ∀ {head : List Expr} {start other : Nat}
    {recipes tail : List Expr},
    recipeConstraints start recipes = recipeConstraints other head ++ tail →
      (head ≠ [] → start = other) ∧
        ∃ rest, recipes = head ++ rest ∧ tail = recipeConstraints (start + head.length) rest
  | [], _, _, recipes, _, rows => ⟨fun absent => absurd rfl absent, recipes, rfl, rows.symm⟩
  | _ :: _, _, _, [], _, rows => nomatch rows
  | wanted :: head, start, other, recipe :: recipes, tail, rows => by
      injection rows with row rows
      injection row with column negated
      injection column with column
      injection negated with _ same
      obtain ⟨_, rest, split, later⟩ := recipeConstraints_split rows
      refine ⟨fun _ => column, rest, by rw [same, split]; rfl, ?_⟩
      rw [later, List.length_cons, Nat.add_right_comm, Nat.add_assoc]

namespace Witness

def execute (env : Env) (batches : List WitnessBatch) : Env :=
  batches.foldl WitnessBatch.execute env

def Computed (env : Env) (batches : List WitnessBatch) : Prop :=
  ∀ batch ∈ batches, batch.Computed env

def Ordered : Nat → List WitnessBatch → Nat → Prop
  | start, [], finish => start ≤ finish
  | start, batch :: rest, finish =>
      start ≤ batch.start ∧ batch.Causal ∧ Ordered batch.finish rest finish

@[simp] theorem execute_append (env : Env) (first second : List WitnessBatch) :
    execute env (first ++ second) = execute (execute env first) second :=
  List.foldl_append ..

@[simp] theorem computed_append (env : Env) (first second : List WitnessBatch) :
    Computed env (first ++ second) ↔ Computed env first ∧ Computed env second := by
  simp only [Computed, List.mem_append]
  exact ⟨fun all => ⟨fun batch member => all batch (Or.inl member),
      fun batch member => all batch (Or.inr member)⟩,
    fun both batch member => member.elim (both.1 batch) (both.2 batch)⟩

theorem Ordered.le : ∀ {start : Nat} {batches : List WitnessBatch} {finish : Nat},
    Ordered start batches finish → start ≤ finish
  | _, [], _, ordered => ordered
  | _, _ :: _, _, ordered =>
      le_trans ordered.1 (le_trans (Nat.le_add_right _ _) (Ordered.le ordered.2.2))

theorem Ordered.mono : ∀ {start lower : Nat} {batches : List WitnessBatch} {finish upper : Nat},
    Ordered start batches finish → lower ≤ start → finish ≤ upper → Ordered lower batches upper
  | _, _, [], _, _, ordered, below, above => le_trans below (le_trans ordered above)
  | _, _, _ :: _, _, _, ordered, below, above =>
      ⟨le_trans below ordered.1, ordered.2.1, Ordered.mono ordered.2.2 le_rfl above⟩

theorem Ordered.append : ∀ {start : Nat} {first : List WitnessBatch} {middle : Nat}
    {second : List WitnessBatch} {finish : Nat},
    Ordered start first middle → Ordered middle second finish →
      Ordered start (first ++ second) finish
  | _, [], _, _, _, left, right => Ordered.mono right left le_rfl
  | _, _ :: _, _, _, _, left, right => ⟨left.1, left.2.1, Ordered.append left.2.2 right⟩

theorem execute_agrees_below {start : Nat} {batches : List WitnessBatch} {finish : Nat}
    (ordered : Ordered start batches finish) (env : Env) (index : Nat) (below : index < start) :
    execute env batches index = env index := by
  induction batches generalizing env start with
  | nil => rfl
  | cons batch rest ih =>
      have inBatch : index < batch.start := lt_of_lt_of_le below ordered.1
      exact (ih ordered.2.2 (batch.execute env)
        (lt_of_lt_of_le inBatch (Nat.le_add_right _ _))).trans
          (batch.execute_agrees_below env index inBatch)

theorem execute_agrees_above {start : Nat} {batches : List WitnessBatch} {finish : Nat}
    (ordered : Ordered start batches finish) (env : Env) (index : Nat) (above : finish ≤ index) :
    execute env batches index = env index := by
  induction batches generalizing env start with
  | nil => rfl
  | cons batch rest ih =>
      exact (ih ordered.2.2 (batch.execute env)).trans
        (batch.execute_agrees_above env index (le_trans (Ordered.le ordered.2.2) above))

theorem execute_agreesOutside {start : Nat} {batches : List WitnessBatch} {count : Nat}
    (ordered : Ordered start batches (start + count)) (env : Env) :
    AgreesOutside env (execute env batches) start count := fun index outside =>
  outside.elim (execute_agrees_below ordered env index) (execute_agrees_above ordered env index)

theorem execute_computed {start : Nat} {batches : List WitnessBatch} {finish : Nat}
    (ordered : Ordered start batches finish) (env : Env) :
    Computed (execute env batches) batches := by
  induction batches generalizing env start with
  | nil => exact fun _ member => absurd member List.not_mem_nil
  | cons batch rest ih =>
      intro current member
      rcases List.mem_cons.mp member with rfl | later
      · exact WitnessBatch.computed_of_agree ordered.2.1
          (execute_agrees_below ordered.2.2 (current.execute env))
          (current.execute_computed env ordered.2.1)
      · exact ih ordered.2.2 (batch.execute env) current later

theorem execute_agreeOff {dead : Nat → Prop} {batches : List WitnessBatch}
    (avoids : ∀ batch ∈ batches, batch.Avoids dead) {left right : Env}
    (agree : AgreeOff dead left right) :
    AgreeOff dead (execute left batches) (execute right batches) := by
  induction batches generalizing left right with
  | nil => exact agree
  | cons batch rest ih =>
      exact ih (fun later member => avoids later (by simp [member]))
        (WitnessBatch.execute_agreeOff (avoids batch (by simp)) agree)

theorem agreeOff_fill {dead : Nat → Prop} {start finish : Nat} {inner : List WitnessBatch}
    (ordered : Ordered start inner finish) {fill : Env → Env} {left right : Env}
    (outside : ∀ column, column < start ∨ finish ≤ column → fill right column = right column)
    (live : ∀ column, start ≤ column → column < finish → ¬dead column →
      execute left inner column = fill right column)
    (agree : AgreeOff dead left right) : AgreeOff dead (execute left inner) (fill right) := by
  intro column alive
  by_cases inside : start ≤ column ∧ column < finish
  · exact live column inside.1 inside.2 alive
  · have out : column < start ∨ finish ≤ column := by omega
    rw [outside column out, ← agree column alive]
    exact out.elim (execute_agrees_below ordered left column)
      (execute_agrees_above ordered left column)

theorem execute_of_rows : ∀ (batches : List WitnessBatch) (env : Env) (start : Nat)
    (recipes : List Expr),
    (∀ batch ∈ batches, batch.hints = [] ∧ batch.recipes ≠ []) →
    recipeConstraints start recipes =
        batches.flatMap (fun batch => recipeConstraints batch.start batch.recipes) →
      executeRecipes env start recipes = execute env batches
  | [], _, _, [], _, _ => rfl
  | [], _, _, _ :: _, _, rows => nomatch rows
  | batch :: rest, env, start, recipes, plain, rows => by
      obtain ⟨same, tail, split, later⟩ := recipeConstraints_split (rows.trans List.flatMap_cons)
      have first := plain batch (by simp)
      have run : batch.execute env = executeRecipes env start batch.recipes := by
        unfold WitnessBatch.execute
        rw [first.1, same first.2]
        rfl
      rw [split, executeRecipes_append, ← run]
      exact execute_of_rows rest _ _ tail (fun later member => plain later (by simp [member]))
        later.symm

theorem witnesses_append (left right : List Op) :
    witnesses (left ++ right) = witnesses left ++ witnesses right :=
  List.flatMap_append

theorem witnesses_cons (operation : Op) (rest : List Op) :
    witnesses (operation :: rest) = operation.witnesses ++ witnesses rest :=
  List.flatMap_cons

theorem witnesses_assertions (rows : List Expr) : witnesses (rows.map Op.assertZero) = [] := by
  induction rows with
  | nil => rfl
  | cons row rest ih => rw [List.map_cons, witnesses_cons, ih]; rfl

theorem computed_holdsFlat {env : Env} {ops : List Op} (only : ∀ op ∈ ops, op.RecipesOnly)
    (computed : Computed env (witnesses ops)) : holdsFlat env ops := by
  intro row member
  obtain ⟨op, opMember, rowMember⟩ := List.mem_flatMap.mp member
  cases op with
  | witness batch =>
      exact (computed batch (List.mem_flatMap.mpr ⟨_, opMember, List.mem_singleton.mpr rfl⟩)).1
        row rowMember
  | mark _ => exact absurd rowMember List.not_mem_nil
  | assertZero _ => exact False.elim (only _ opMember)
  | subcircuit _ => exact False.elim (only _ opMember)

theorem ordered_batch {offset : Nat} {recipes : List Expr} {ops : List Op}
    (batches : witnesses ops = [WitnessBatch.arithmetic offset recipes])
    (count : localLength ops = recipes.length) (causal : RecipesCausal offset recipes) :
    Ordered offset (witnesses ops) (offset + localLength ops) := by
  rw [batches, count]
  exact ⟨le_rfl, ⟨causal, trivial⟩, by simp [Ordered, WitnessBatch.finish]⟩

theorem computed_batch {env : Env} {offset : Nat} {recipes : List Expr} {ops : List Op}
    (batches : witnesses ops = [WitnessBatch.arithmetic offset recipes])
    (computed : Computed env (witnesses ops)) :
    ConstraintsHold env (recipeConstraints offset recipes) :=
  (computed (WitnessBatch.arithmetic offset recipes) (by simp [batches])).1

theorem complete {env : Env} {offset : Nat} {ops : List Op}
    (ordered : Ordered offset (witnesses ops) (offset + localLength ops))
    (rows : ∀ completed, (∀ index, index < offset → completed index = env index) →
      Computed completed (witnesses ops) → holdsFlat completed ops) :
    ∃ completed, AgreesOutside env completed offset (localLength ops) ∧ holdsFlat completed ops :=
  ⟨execute env (witnesses ops), execute_agreesOutside ordered env,
    rows _ (execute_agrees_below ordered env) (execute_computed ordered env)⟩

end Witness

end NightstreamFPrime.Circuit
