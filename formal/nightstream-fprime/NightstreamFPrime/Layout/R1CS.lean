import NightstreamFPrime.Circuit.Basic

/-!
Owns the physical R1CS lowering of logical circuit expressions. Every
multiplication allocates one fresh physical variable and one row. Every
logical zero assertion adds one final row. The proof is structural in the
expression syntax and does not evaluate an emitted artifact in the kernel.
-/

namespace NightstreamFPrime.Layout.R1CS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- Sparse affine form over physical variable indices. -/
structure LinearCombination where
  constant : F
  terms : List (Nat × F)
deriving Repr, DecidableEq

namespace LinearCombination

def eval (env : Env) (combination : LinearCombination) : F :=
  combination.constant +
    (combination.terms.map fun term => term.2 * env term.1).sum

def zero : LinearCombination := ⟨0, []⟩
def one : LinearCombination := ⟨1, []⟩
def const (value : F) : LinearCombination := ⟨value, []⟩
def ofVar (index : Nat) : LinearCombination := ⟨0, [(index, 1)]⟩

def add (left right : LinearCombination) : LinearCombination :=
  ⟨left.constant + right.constant, left.terms ++ right.terms⟩

def scale (coefficient : F) (combination : LinearCombination) :
    LinearCombination :=
  ⟨coefficient * combination.constant,
    combination.terms.map fun term => (term.1, coefficient * term.2)⟩

@[simp] theorem eval_zero (env : Env) : zero.eval env = 0 := rfl
@[simp] theorem eval_one (env : Env) : one.eval env = 1 := rfl
@[simp] theorem eval_const (env : Env) (value : F) :
    (const value).eval env = value := by
  simp [const, eval]
@[simp] theorem eval_ofVar (env : Env) (index : Nat) :
    (ofVar index).eval env = env index := by
  simp [ofVar, eval]

@[simp] theorem eval_add (env : Env) (left right : LinearCombination) :
    (add left right).eval env = left.eval env + right.eval env := by
  simp [add, eval, List.sum_append, add_assoc, add_left_comm]

@[simp] theorem eval_scale (env : Env) (coefficient : F)
    (combination : LinearCombination) :
    (scale coefficient combination).eval env =
      coefficient * combination.eval env := by
  have sumScale :
      (combination.terms.map (fun term =>
        (coefficient * term.2) * env term.1)).sum =
        coefficient *
          (combination.terms.map (fun term => term.2 * env term.1)).sum := by
    induction combination.terms with
    | nil => simp
    | cons term rest ih =>
        simp only [List.map_cons, List.sum_cons]
        rw [ih, mul_add, mul_assoc]
  unfold scale eval
  rw [List.map_map]
  change coefficient * combination.constant +
      (combination.terms.map (fun term =>
        (coefficient * term.2) * env term.1)).sum =
    coefficient * (combination.constant +
      (combination.terms.map (fun term => term.2 * env term.1)).sum)
  rw [sumScale, mul_add]

end LinearCombination

/-- One physical rank-one equation `A · B = C`. -/
structure Row where
  a : LinearCombination
  b : LinearCombination
  c : LinearCombination
deriving Repr, DecidableEq

def Row.Holds (env : Env) (row : Row) : Prop :=
  row.a.eval env * row.b.eval env = row.c.eval env

def RowsHold (env : Env) (rows : List Row) : Prop :=
  ∀ row ∈ rows, row.Holds env

theorem rowsHold_append (env : Env) (first second : List Row) :
    RowsHold env (first ++ second) ↔ RowsHold env first ∧ RowsHold env second := by
  constructor
  · intro holds
    exact ⟨
      fun row member => holds row (List.mem_append_left second member),
      fun row member => holds row (List.mem_append_right first member)⟩
  · rintro ⟨firstHolds, secondHolds⟩ row member
    rcases List.mem_append.mp member with member | member
    · exact firstHolds row member
    · exact secondHolds row member

/-- Number of fresh multiplication variables needed by one expression. -/
def mulCount : Expr → Nat
  | .var _ | .const _ => 0
  | .add left right => mulCount left + mulCount right
  | .mul left right => mulCount left + mulCount right + 1

/-- Lowering result for one expression. -/
structure LoweredExpression where
  value : LinearCombination
  next : Nat
  rows : List Row

def lowerExpression : Expr → Nat → LoweredExpression
  | .var index, start => ⟨LinearCombination.ofVar index, start, []⟩
  | .const value, start => ⟨LinearCombination.const value, start, []⟩
  | .add left right, start =>
      let loweredLeft := lowerExpression left start
      let loweredRight := lowerExpression right loweredLeft.next
      ⟨LinearCombination.add loweredLeft.value loweredRight.value,
        loweredRight.next, loweredLeft.rows ++ loweredRight.rows⟩
  | .mul left right, start =>
      let loweredLeft := lowerExpression left start
      let loweredRight := lowerExpression right loweredLeft.next
      let output := loweredRight.next
      let row : Row :=
        ⟨loweredLeft.value, loweredRight.value,
          LinearCombination.ofVar output⟩
      ⟨LinearCombination.ofVar output, output + 1,
        loweredLeft.rows ++ loweredRight.rows ++ [row]⟩

@[simp] theorem lowerExpression_next (expression : Expr) (start : Nat) :
    (lowerExpression expression start).next = start + mulCount expression := by
  induction expression generalizing start with
  | var index => simp [lowerExpression, mulCount]
  | const value => simp [lowerExpression, mulCount]
  | add left right leftIH rightIH =>
      simp [lowerExpression, mulCount, leftIH, rightIH]
      omega
  | mul left right leftIH rightIH =>
      simp [lowerExpression, mulCount, leftIH, rightIH]
      omega

@[simp] theorem lowerExpression_rows_length (expression : Expr) (start : Nat) :
    (lowerExpression expression start).rows.length = mulCount expression := by
  induction expression generalizing start with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp [lowerExpression, mulCount, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp [lowerExpression, mulCount, leftIH, rightIH]
      omega

theorem lowerExpression_sound (env : Env) (expression : Expr) (start : Nat)
    (rows : RowsHold env (lowerExpression expression start).rows) :
    (lowerExpression expression start).value.eval env = expression.eval env := by
  induction expression generalizing start with
  | var index => simp [lowerExpression]
  | const value => simp [lowerExpression]
  | add left right leftIH rightIH =>
      let loweredLeft := lowerExpression left start
      let loweredRight := lowerExpression right loweredLeft.next
      have separated := (rowsHold_append env loweredLeft.rows loweredRight.rows).mp (by
        simpa [lowerExpression, loweredLeft, loweredRight] using rows)
      calc
        (lowerExpression (left + right) start).value.eval env =
            loweredLeft.value.eval env + loweredRight.value.eval env := by
          simp [lowerExpression, loweredLeft, loweredRight]
        _ = left.eval env + right.eval env := by
          rw [leftIH start separated.1, rightIH loweredLeft.next separated.2]
        _ = (left + right).eval env := rfl
  | mul left right leftIH rightIH =>
      let loweredLeft := lowerExpression left start
      let loweredRight := lowerExpression right loweredLeft.next
      let output := loweredRight.next
      let productRow : Row :=
        ⟨loweredLeft.value, loweredRight.value,
          LinearCombination.ofVar output⟩
      have firstSplit := (rowsHold_append env loweredLeft.rows
        (loweredRight.rows ++ [productRow])).mp (by
          simpa [lowerExpression, loweredLeft, loweredRight, output, productRow]
            using rows)
      have secondSplit :=
        (rowsHold_append env loweredRight.rows [productRow]).mp firstSplit.2
      have productHolds : productRow.Holds env :=
        secondSplit.2 productRow (by simp)
      have productEquation :
          loweredLeft.value.eval env * loweredRight.value.eval env =
            env output := by
        simpa [Row.Holds, productRow] using productHolds
      calc
        (lowerExpression (left * right) start).value.eval env = env output := by
          simp [lowerExpression, loweredLeft, loweredRight, output]
        _ = loweredLeft.value.eval env * loweredRight.value.eval env := by
          exact productEquation.symm
        _ = left.eval env * right.eval env := by
          rw [leftIH start firstSplit.1,
            rightIH loweredLeft.next secondSplit.1]
        _ = (left * right).eval env := rfl

/-- A proof-producing recognition result for an affine logical expression. -/
structure AffineResult (expression : Expr) where
  combination : LinearCombination
  sound : ∀ env, combination.eval env = expression.eval env

/-- Recognize exactly the affine fragment used by staged witness recipes.
Multiplication by a field constant is coefficient scaling, not a rank-one
row. -/
def lowerAffine : (expression : Expr) → Option (AffineResult expression)
  | .var index =>
      some ⟨LinearCombination.ofVar index, by intro env; simp⟩
  | .const value =>
      some ⟨LinearCombination.const value, by intro env; simp⟩
  | .add left right =>
      match lowerAffine left, lowerAffine right with
      | some loweredLeft, some loweredRight =>
          some ⟨LinearCombination.add loweredLeft.combination
            loweredRight.combination, by
              intro env
              rw [LinearCombination.eval_add, loweredLeft.sound env,
                loweredRight.sound env]
              rfl⟩
      | _, _ => none
  | .mul (.const coefficient) right =>
      match lowerAffine right with
      | some loweredRight =>
          some ⟨LinearCombination.scale coefficient loweredRight.combination, by
            intro env
            rw [LinearCombination.eval_scale, loweredRight.sound env]
            rfl⟩
      | none => none
  | .mul left (.const coefficient) =>
      match lowerAffine left with
      | some loweredLeft =>
          some ⟨LinearCombination.scale coefficient loweredLeft.combination, by
            intro env
            rw [LinearCombination.eval_scale, loweredLeft.sound env, mul_comm]
            rfl⟩
      | none => none
  | .mul _ _ => none

/-- Syntactic affine recognition succeeded. -/
def IsAffine (expression : Expr) : Prop :=
  ∃ lowered, lowerAffine expression = some lowered

@[simp] theorem isAffine_var (index : Nat) : IsAffine (.var index) := by
  simp [IsAffine, lowerAffine]

@[simp] theorem isAffine_const (value : F) : IsAffine (.const value) := by
  simp [IsAffine, lowerAffine]

theorem IsAffine.add {left right : Expr}
    (leftAffine : IsAffine left) (rightAffine : IsAffine right) :
    IsAffine (left + right) := by
  rcases leftAffine with ⟨loweredLeft, leftEquals⟩
  rcases rightAffine with ⟨loweredRight, rightEquals⟩
  simp [IsAffine, lowerAffine, leftEquals, rightEquals]

theorem IsAffine.const_mul (coefficient : F) {expression : Expr}
    (affine : IsAffine expression) :
    IsAffine (.const coefficient * expression) := by
  rcases affine with ⟨lowered, equals⟩
  simp [IsAffine, lowerAffine, equals]

/-- One direct row and its exact canonical recipe-equation proof. -/
structure RecipeRowResult (output : Nat) (recipe : Expr) where
  row : Row
  sound : ∀ env, row.Holds env →
    (Expr.var output - recipe).eval env = 0
  complete : ∀ env, (Expr.var output - recipe).eval env = 0 →
    row.Holds env

def affineRecipeRow (output : Nat) (recipe : Expr)
    (lowered : AffineResult recipe) : RecipeRowResult output recipe where
  row := ⟨lowered.combination, LinearCombination.one,
    LinearCombination.ofVar output⟩
  sound := by
    intro env holds
    have equation : recipe.eval env = env output := by
      rw [← lowered.sound env]
      simpa [Row.Holds] using holds
    simp [Expr.eval_sub, equation]
  complete := by
    intro env constraint
    have equation : env output = recipe.eval env :=
      sub_eq_zero.mp (by simpa only [Expr.eval_sub] using constraint)
    simpa [Row.Holds, lowered.sound env] using equation.symm

def quadraticRecipeRow (output : Nat) (left right : Expr)
    (loweredLeft : AffineResult left) (loweredRight : AffineResult right) :
    RecipeRowResult output (.mul left right) where
  row := ⟨loweredLeft.combination, loweredRight.combination,
    LinearCombination.ofVar output⟩
  sound := by
    intro env holds
    have equation : left.eval env * right.eval env = env output := by
      simpa [Row.Holds, loweredLeft.sound env, loweredRight.sound env] using holds
    simp [Expr.eval_sub, equation]
  complete := by
    intro env constraint
    have equation : env output = left.eval env * right.eval env :=
      sub_eq_zero.mp (by simpa only [Expr.eval_sub] using constraint)
    simpa [Row.Holds, loweredLeft.sound env, loweredRight.sound env] using
      equation.symm

/-- Compile an affine or one-rank quadratic recipe directly against its
already allocated logical witness variable. -/
def directRecipeRow (output : Nat) (recipe : Expr) :
    Option (RecipeRowResult output recipe) :=
  match lowerAffine recipe with
  | some lowered => some (affineRecipeRow output recipe lowered)
  | none =>
      match recipe with
      | .mul left right =>
          match lowerAffine left, lowerAffine right with
          | some loweredLeft, some loweredRight =>
              some (quadraticRecipeRow output left right
                loweredLeft loweredRight)
          | _, _ => none
      | _ => none

/-- One recipe is affine or one rank-one product of affine expressions. -/
def IsDirectRecipe (output : Nat) (recipe : Expr) : Prop :=
  ∃ lowered, directRecipeRow output recipe = some lowered

theorem IsDirectRecipe.of_affine (output : Nat) {recipe : Expr}
    (affine : IsAffine recipe) : IsDirectRecipe output recipe := by
  rcases affine with ⟨lowered, equals⟩
  simp [IsDirectRecipe, directRecipeRow, equals]

theorem IsDirectRecipe.mul (output : Nat) {left right : Expr}
    (leftAffine : IsAffine left) (rightAffine : IsAffine right) :
    IsDirectRecipe output (left * right) := by
  rcases leftAffine with ⟨loweredLeft, leftEquals⟩
  rcases rightAffine with ⟨loweredRight, rightEquals⟩
  cases productEquals : lowerAffine (left * right) with
  | none =>
      simp [IsDirectRecipe, directRecipeRow, productEquals,
        leftEquals, rightEquals]
  | some lowered =>
      simp [IsDirectRecipe, directRecipeRow, productEquals]

/-- Every recipe in a batch has a one-row lowering at its exact output
offset. -/
def RecipesDirect : Nat → List Expr → Prop
  | _, [] => True
  | output, recipe :: rest =>
      IsDirectRecipe output recipe ∧ RecipesDirect (output + 1) rest

theorem recipesDirect_append (output : Nat) (first second : List Expr)
    (firstDirect : RecipesDirect output first)
    (secondDirect : RecipesDirect (output + first.length) second) :
    RecipesDirect output (first ++ second) := by
  induction first generalizing output with
  | nil => simpa [RecipesDirect] using secondDirect
  | cons recipe rest ih =>
      constructor
      · exact firstDirect.1
      · apply ih (output + 1) firstDirect.2
        convert secondDirect using 1 <;> simp <;> omega

/-- One recognized canonical logical constraint and its direct physical row. -/
structure DirectConstraintResult (expression : Expr) where
  row : Row
  sound : ∀ env, row.Holds env → expression.eval env = 0
  complete : ∀ env, expression.eval env = 0 → row.Holds env

def affineConstraintRow (expression : Expr) (lowered : AffineResult expression) :
    DirectConstraintResult expression where
  row := ⟨lowered.combination, LinearCombination.one,
    LinearCombination.zero⟩
  sound := by
    intro env holds
    simpa [Row.Holds, lowered.sound env] using holds
  complete := by
    intro env holds
    simpa [Row.Holds, lowered.sound env] using holds

def affineConstraint (expression : Expr) :
    Option (DirectConstraintResult expression) :=
  match lowerAffine expression with
  | some lowered => some (affineConstraintRow expression lowered)
  | none => none

/-- Recognize only `recipeConstraints` equations. Other zero constraints use
the general expression lowering below. -/
def directConstraint : (expression : Expr) →
    Option (DirectConstraintResult expression)
  | .add (.var output) (.mul (.const coefficient) recipe) =>
      if coefficientEquals : coefficient = -1 then
        match directRecipeRow output recipe with
        | some lowered =>
            some ⟨lowered.row, by
              intro env holds
              rw [coefficientEquals]
              simpa [Expr.eval, neg_one_mul, sub_eq_add_neg] using
                lowered.sound env holds, by
              intro env constraint
              apply lowered.complete env
              rw [coefficientEquals] at constraint
              simpa [Expr.eval, neg_one_mul, sub_eq_add_neg] using constraint⟩
        | none => affineConstraint _
      else
        affineConstraint _
  | expression => affineConstraint expression

/-- Lowering result for one logical zero constraint. -/
structure LoweredConstraint where
  next : Nat
  rows : List Row

def lowerGenericConstraint (expression : Expr) (start : Nat) :
    LoweredConstraint :=
  let lowered := lowerExpression expression start
  let assertion : Row := ⟨lowered.value, LinearCombination.one,
    LinearCombination.zero⟩
  ⟨lowered.next, lowered.rows ++ [assertion]⟩

@[simp] theorem lowerGenericConstraint_next (expression : Expr) (start : Nat) :
    (lowerGenericConstraint expression start).next =
      start + mulCount expression := by
  simp [lowerGenericConstraint]

@[simp] theorem lowerGenericConstraint_rows_length (expression : Expr)
    (start : Nat) :
    (lowerGenericConstraint expression start).rows.length =
      mulCount expression + 1 := by
  simp [lowerGenericConstraint]

theorem lowerGenericConstraint_sound (env : Env) (expression : Expr)
    (start : Nat)
    (rows : RowsHold env (lowerGenericConstraint expression start).rows) :
    expression.eval env = 0 := by
  let lowered := lowerExpression expression start
  let assertion : Row := ⟨lowered.value, LinearCombination.one,
    LinearCombination.zero⟩
  have separated := (rowsHold_append env lowered.rows [assertion]).mp (by
    simpa [lowerGenericConstraint, lowered, assertion] using rows)
  have assertionHolds : assertion.Holds env :=
    separated.2 assertion (by simp)
  have loweredSound := lowerExpression_sound env expression start separated.1
  calc
    expression.eval env = lowered.value.eval env := loweredSound.symm
    _ = 0 := by
      simpa [Row.Holds, assertion] using assertionHolds

def lowerConstraint (expression : Expr) (start : Nat) : LoweredConstraint :=
  match directConstraint expression with
  | some direct => ⟨start, [direct.row]⟩
  | none => lowerGenericConstraint expression start

/-- Fresh intermediate columns allocated by one optimized constraint. -/
def constraintFreshCount (expression : Expr) : Nat :=
  match directConstraint expression with
  | some _ => 0
  | none => mulCount expression

/-- Physical rows allocated by one optimized constraint. -/
def constraintRowCount (expression : Expr) : Nat :=
  match directConstraint expression with
  | some _ => 1
  | none => mulCount expression + 1

theorem directConstraint_recipe_of_direct (output : Nat) (recipe : Expr)
    (direct : IsDirectRecipe output recipe) :
    ∃ lowered,
      directConstraint (Expr.var output - recipe) = some lowered := by
  rcases direct with ⟨lowered, equals⟩
  change ∃ result,
    directConstraint (.add (.var output) (.mul (.const (-1)) recipe)) =
      some result
  simp [directConstraint, equals]

theorem constraintFreshCount_recipe_eq_zero (output : Nat) (recipe : Expr)
    (direct : IsDirectRecipe output recipe) :
    constraintFreshCount (Expr.var output - recipe) = 0 := by
  rcases directConstraint_recipe_of_direct output recipe direct with
    ⟨lowered, equals⟩
  simp [constraintFreshCount, equals]

theorem constraintRowCount_recipe_eq_one (output : Nat) (recipe : Expr)
    (direct : IsDirectRecipe output recipe) :
    constraintRowCount (Expr.var output - recipe) = 1 := by
  rcases directConstraint_recipe_of_direct output recipe direct with
    ⟨lowered, equals⟩
  simp [constraintRowCount, equals]

@[simp] theorem lowerConstraint_next (expression : Expr) (start : Nat) :
    (lowerConstraint expression start).next =
      start + constraintFreshCount expression := by
  cases result : directConstraint expression with
  | none => simp [lowerConstraint, constraintFreshCount, result]
  | some direct => simp [lowerConstraint, constraintFreshCount, result]

@[simp] theorem lowerConstraint_rows_length (expression : Expr) (start : Nat) :
    (lowerConstraint expression start).rows.length =
      constraintRowCount expression := by
  cases result : directConstraint expression with
  | none => simp [lowerConstraint, constraintRowCount, result]
  | some direct => simp [lowerConstraint, constraintRowCount, result]

theorem lowerConstraint_sound (env : Env) (expression : Expr) (start : Nat)
    (rows : RowsHold env (lowerConstraint expression start).rows) :
    expression.eval env = 0 := by
  cases result : directConstraint expression with
  | none =>
      exact lowerGenericConstraint_sound env expression start (by
        simpa [lowerConstraint, result] using rows)
  | some direct =>
      apply direct.sound env
      exact rows direct.row (by simp [lowerConstraint, result])

theorem lowerGenericConstraint_complete_of_mulCount_zero
    (env : Env) (expression : Expr) (start : Nat)
    (count : mulCount expression = 0) (holds : expression.eval env = 0) :
    RowsHold env (lowerGenericConstraint expression start).rows := by
  let lowered := lowerExpression expression start
  let assertion : Row := ⟨lowered.value, LinearCombination.one,
    LinearCombination.zero⟩
  have rowsEmpty : lowered.rows = [] := by
    have lengthZero : lowered.rows.length = 0 := by
      simpa [lowered, count] using lowerExpression_rows_length expression start
    cases rowsEquals : lowered.rows with
    | nil => rfl
    | cons row rest => simp [rowsEquals] at lengthZero
  have emptyRowsHold : RowsHold env lowered.rows := by
    rw [rowsEmpty]
    intro row member
    simp at member
  have loweredSound : lowered.value.eval env = expression.eval env :=
    lowerExpression_sound env expression start emptyRowsHold
  intro row member
  have rowEquals : row = assertion := by
    simpa [lowerGenericConstraint, lowered, assertion, rowsEmpty] using member
  subst row
  simp [Row.Holds, assertion, loweredSound, holds]

theorem lowerConstraint_complete_of_freshCount_zero
    (env : Env) (expression : Expr) (start : Nat)
    (fresh : constraintFreshCount expression = 0)
    (holds : expression.eval env = 0) :
    RowsHold env (lowerConstraint expression start).rows := by
  cases result : directConstraint expression with
  | none =>
      have complete := lowerGenericConstraint_complete_of_mulCount_zero
        env expression start (by
          simpa [constraintFreshCount, result] using fresh) holds
      simpa [lowerConstraint, result] using complete
  | some direct =>
      intro row member
      have rowEquals : row = direct.row := by
        simpa [lowerConstraint, result] using member
      subst row
      exact direct.complete env holds

/-- Lowering result for an ordered logical constraint list. -/
structure LoweredConstraints where
  next : Nat
  rows : List Row

def lowerConstraints : List Expr → Nat → LoweredConstraints
  | [], start => ⟨start, []⟩
  | expression :: rest, start =>
      let first := lowerConstraint expression start
      let tail := lowerConstraints rest first.next
      ⟨tail.next, first.rows ++ tail.rows⟩

def totalFreshCount (constraints : List Expr) : Nat :=
  (constraints.map constraintFreshCount).sum

def totalRowCount (constraints : List Expr) : Nat :=
  (constraints.map constraintRowCount).sum

@[simp] theorem totalFreshCount_append (first second : List Expr) :
    totalFreshCount (first ++ second) =
      totalFreshCount first + totalFreshCount second := by
  simp [totalFreshCount, List.map_append, List.sum_append]

@[simp] theorem totalRowCount_append (first second : List Expr) :
    totalRowCount (first ++ second) =
      totalRowCount first + totalRowCount second := by
  simp [totalRowCount, List.map_append, List.sum_append]

theorem recipeConstraints_totalFreshCount (output : Nat)
    (recipes : List Expr) (direct : RecipesDirect output recipes) :
    totalFreshCount (recipeConstraints output recipes) = 0 := by
  induction recipes generalizing output with
  | nil => rfl
  | cons recipe rest ih =>
      simp only [recipeConstraints, totalFreshCount, List.map_cons,
        List.sum_cons]
      have tail := ih (output + 1) direct.2
      unfold totalFreshCount at tail
      rw [constraintFreshCount_recipe_eq_zero output recipe direct.1, tail]

theorem recipeConstraints_noFresh (output : Nat) (recipes : List Expr)
    (direct : RecipesDirect output recipes) :
    ∀ expression ∈ recipeConstraints output recipes,
      constraintFreshCount expression = 0 := by
  induction recipes generalizing output with
  | nil => simp [recipeConstraints]
  | cons recipe rest ih =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · exact constraintFreshCount_recipe_eq_zero output recipe direct.1
      · exact ih (output + 1) direct.2 expression member

theorem recipeConstraints_rowsOne (output : Nat) (recipes : List Expr)
    (direct : RecipesDirect output recipes) :
    ∀ expression ∈ recipeConstraints output recipes,
      constraintRowCount expression = 1 := by
  induction recipes generalizing output with
  | nil => simp [recipeConstraints]
  | cons recipe rest ih =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · exact constraintRowCount_recipe_eq_one output recipe direct.1
      · exact ih (output + 1) direct.2 expression member

theorem recipeConstraints_totalRowCount (output : Nat)
    (recipes : List Expr) (direct : RecipesDirect output recipes) :
    totalRowCount (recipeConstraints output recipes) = recipes.length := by
  induction recipes generalizing output with
  | nil => rfl
  | cons recipe rest ih =>
      simp only [recipeConstraints, totalRowCount, List.map_cons,
        List.sum_cons, List.length_cons]
      have tail := ih (output + 1) direct.2
      unfold totalRowCount at tail
      rw [constraintRowCount_recipe_eq_one output recipe direct.1, tail]
      omega

theorem totalFreshCount_eq_zero_of_noFresh (constraints : List Expr)
    (noFresh : ∀ expression ∈ constraints,
      constraintFreshCount expression = 0) :
    totalFreshCount constraints = 0 := by
  induction constraints with
  | nil => rfl
  | cons expression rest ih =>
      have tail := ih (by
        intro current member
        exact noFresh current (by simp [member]))
      unfold totalFreshCount at tail
      simp only [totalFreshCount, List.map_cons, List.sum_cons]
      rw [noFresh expression (by simp), tail]

theorem totalRowCount_eq_length_of_rowsOne (constraints : List Expr)
    (rowsOne : ∀ expression ∈ constraints,
      constraintRowCount expression = 1) :
    totalRowCount constraints = constraints.length := by
  induction constraints with
  | nil => rfl
  | cons expression rest ih =>
      have tail := ih (by
        intro current member
        exact rowsOne current (by simp [member]))
      unfold totalRowCount at tail
      simp only [totalRowCount, List.map_cons, List.sum_cons, List.length_cons]
      rw [rowsOne expression (by simp), tail]
      omega

@[simp] theorem lowerConstraints_next (constraints : List Expr) (start : Nat) :
    (lowerConstraints constraints start).next =
      start + totalFreshCount constraints := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest ih =>
      simp [lowerConstraints, totalFreshCount, ih]
      omega

theorem lowerConstraints_append_rows (first second : List Expr) (start : Nat) :
    (lowerConstraints (first ++ second) start).rows =
      (lowerConstraints first start).rows ++
        (lowerConstraints second
          (start + totalFreshCount first)).rows := by
  induction first generalizing start with
  | nil => simp [lowerConstraints, totalFreshCount]
  | cons expression rest inductionHypothesis =>
      simp [lowerConstraints, totalFreshCount, inductionHypothesis,
        List.append_assoc, Nat.add_assoc]

@[simp] theorem lowerConstraints_rows_length (constraints : List Expr)
    (start : Nat) :
    (lowerConstraints constraints start).rows.length =
      totalRowCount constraints := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest ih =>
      simp [lowerConstraints, totalRowCount, ih]

theorem lowerConstraints_sound (env : Env) (constraints : List Expr)
    (start : Nat)
    (rows : RowsHold env (lowerConstraints constraints start).rows) :
    ConstraintsHold env constraints := by
  induction constraints generalizing start with
  | nil =>
      intro expression member
      simp at member
  | cons expression rest ih =>
      let first := lowerConstraint expression start
      let tail := lowerConstraints rest first.next
      have separated := (rowsHold_append env first.rows tail.rows).mp (by
        simpa [lowerConstraints, first, tail] using rows)
      intro current member
      simp only [List.mem_cons] at member
      rcases member with equal | member
      · subst current
        exact lowerConstraint_sound env _ start separated.1
      · exact ih first.next separated.2 current member

/-- A logical constraint list whose optimized lowering allocates no extra
columns is physically complete in the same environment. -/
theorem lowerConstraints_complete_of_noFresh
    (env : Env) (constraints : List Expr) (start : Nat)
    (noFresh : ∀ expression ∈ constraints,
      constraintFreshCount expression = 0)
    (holds : ConstraintsHold env constraints) :
    RowsHold env (lowerConstraints constraints start).rows := by
  induction constraints generalizing start with
  | nil =>
      intro row member
      simp [lowerConstraints] at member
  | cons expression rest ih =>
      let first := lowerConstraint expression start
      let tail := lowerConstraints rest first.next
      apply (rowsHold_append env first.rows tail.rows).mpr
      constructor
      · exact lowerConstraint_complete_of_freshCount_zero env expression start
          (noFresh expression (by simp)) (holds expression (by simp))
      · apply ih first.next
        · intro current member
          exact noFresh current (by simp [member])
        · intro current member
          exact holds current (by simp [member])

/-- Opaque physical-lowering boundary. A phase supplies only its logical
constraint list and the first fresh column. The generic theorems below keep
parent proof cost independent of the concrete list length. -/
structure LoweringPlan where
  constraints : List Expr
  firstFresh : Nat

namespace LoweringPlan

def lowering (plan : LoweringPlan) : LoweredConstraints :=
  lowerConstraints plan.constraints plan.firstFresh

def rows (plan : LoweringPlan) : List Row := plan.lowering.rows

def next (plan : LoweringPlan) : Nat := plan.lowering.next

def freshColumnCount (plan : LoweringPlan) : Nat :=
  totalFreshCount plan.constraints

def rowCount (plan : LoweringPlan) : Nat := plan.rows.length

theorem rowCount_eq (plan : LoweringPlan) :
    plan.rowCount = totalRowCount plan.constraints := by
  exact lowerConstraints_rows_length plan.constraints plan.firstFresh

theorem next_eq (plan : LoweringPlan) :
    plan.next = plan.firstFresh + plan.freshColumnCount := by
  exact lowerConstraints_next plan.constraints plan.firstFresh

theorem sound (plan : LoweringPlan) (env : Env)
    (physical : RowsHold env plan.rows) :
    ConstraintsHold env plan.constraints := by
  exact lowerConstraints_sound env plan.constraints plan.firstFresh physical

theorem complete_of_noFresh (plan : LoweringPlan) (env : Env)
    (noFresh : ∀ expression ∈ plan.constraints,
      constraintFreshCount expression = 0)
    (logical : ConstraintsHold env plan.constraints) :
    RowsHold env plan.rows := by
  exact lowerConstraints_complete_of_noFresh env plan.constraints
    plan.firstFresh noFresh logical

end LoweringPlan

/-- Exact physical footprint of one proved logical circuit. A phase parent
uses these certified numbers and never unfolds the child's operations. -/
structure CircuitFootprint (circuit : FormalCircuit) where
  freshColumnCount : Nat → Nat
  physicalRowCount : Nat → Nat
  freshColumnCount_eq : ∀ offset,
    totalFreshCount (flatConstraints (Circuit.ops circuit.main offset)) =
      freshColumnCount offset
  physicalRowCount_eq : ∀ offset,
    totalRowCount (flatConstraints (Circuit.ops circuit.main offset)) =
      physicalRowCount offset

end NightstreamFPrime.Layout.R1CS
