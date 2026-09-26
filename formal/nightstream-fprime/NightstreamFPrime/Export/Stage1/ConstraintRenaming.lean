import NightstreamFPrime.Export.Stage1.CompactRows

/-! Structural column transport for the existing optimized R1CS compiler.
The proof observes compiler results; it does not define another compiler. -/

namespace NightstreamFPrime.Export.Stage1.ConstraintRenaming

open NightstreamFPrime.Circuit NightstreamFPrime.Layout NightstreamFPrime.Spec
open CompactRows

def affine (expression : Expr) : Option R1CS.LinearCombination :=
  (R1CS.lowerAffine expression).map R1CS.AffineResult.combination

private theorem affine_add (left right : Expr) :
    affine (.add left right) =
      match affine left, affine right with
      | some a, some b => some (R1CS.LinearCombination.add a b)
      | _, _ => none := by
  cases hl : R1CS.lowerAffine left <;> cases hr : R1CS.lowerAffine right <;>
    simp [affine, R1CS.lowerAffine, hl, hr]

private theorem affine_mul (left right : Expr) :
    affine (.mul left right) =
      match left, right with
      | .const coefficient, value => (affine value).map (R1CS.LinearCombination.scale coefficient)
      | value, .const coefficient => (affine value).map (R1CS.LinearCombination.scale coefficient)
      | _, _ => none := by
  cases left <;> cases right <;> simp only [affine, R1CS.lowerAffine]
  all_goals first | rfl | split <;> simp_all only [Option.map_some, Option.map_none]

theorem affine_rename (column : Nat → Nat) (expression : Expr) :
    affine (renameExpr column expression) = (affine expression).map (renameCombination column) := by
  induction expression with
  | var index => rfl
  | const value => rfl
  | add left right hl hr =>
    simp only [renameExpr, affine_add, hl, hr]
    cases al : affine left <;> cases ar : affine right <;>
      simp only [Option.map_some, Option.map_none, renameCombination_add]
  | mul left right hl hr =>
    cases left <;> cases right <;>
      simp only [renameExpr, affine_mul] at hl hr ⊢
    all_goals first
      | rfl
      | rw [hl]; simp only [Option.map_map, Function.comp_def, R1CS.mapCombinationColumns_scale]
      | rw [hr]; simp only [Option.map_map, Function.comp_def, R1CS.mapCombinationColumns_scale]

def recipe (output : Nat) (expression : Expr) : Option R1CS.Row :=
  (R1CS.directRecipeRow output expression).map R1CS.RecipeRowResult.row

private theorem recipe_eq (output : Nat) (expression : Expr) :
    recipe output expression =
      match affine expression with
      | some value => some ⟨value, R1CS.LinearCombination.one, R1CS.LinearCombination.ofVar output⟩
      | none => match expression with
        | .mul left right => match affine left, affine right with
          | some a, some b => some ⟨a, b, R1CS.LinearCombination.ofVar output⟩
          | _, _ => none
        | _ => none := by
  cases found : R1CS.lowerAffine expression with
  | some value => simp [recipe, R1CS.directRecipeRow, found, affine, R1CS.affineRecipeRow]
  | none =>
    cases expression <;> simp only [recipe, R1CS.directRecipeRow, found, affine]
    all_goals first | rfl | skip
    rename_i left right
    cases hl : R1CS.lowerAffine left <;> cases hr : R1CS.lowerAffine right <;>
      simp [R1CS.quadraticRecipeRow]

theorem recipe_rename (column : Nat → Nat) (output : Nat) (expression : Expr) :
    recipe (column output) (renameExpr column expression) =
      (recipe output expression).map (renameRow column) := by
  rw [recipe_eq, recipe_eq, affine_rename]
  cases value : affine expression with
  | some value => simp [renameRow, R1CS.mapRowColumns, R1CS.mapCombinationColumns_one,
      R1CS.mapCombinationColumns_ofVar]
  | none =>
    simp only [Option.map_none]
    cases expression <;> simp only [renameExpr]
    all_goals first | rfl | skip
    rw [affine_rename, affine_rename]
    rename_i left right
    cases hl : affine left <;> cases hr : affine right <;>
      simp [renameRow, R1CS.mapRowColumns, R1CS.mapCombinationColumns_ofVar]

def affineConstraint (expression : Expr) : Option R1CS.Row :=
  (R1CS.affineConstraint expression).map R1CS.DirectConstraintResult.row

private theorem affineConstraint_eq (expression : Expr) :
    affineConstraint expression = (affine expression).map (fun value =>
      ⟨value, R1CS.LinearCombination.one, R1CS.LinearCombination.zero⟩) := by
  cases found : R1CS.lowerAffine expression <;>
    simp [affineConstraint, R1CS.affineConstraint, affine, found, R1CS.affineConstraintRow]

theorem affineConstraint_rename (column : Nat → Nat) (expression : Expr) :
    affineConstraint (renameExpr column expression) =
      (affineConstraint expression).map (renameRow column) := by
  rw [affineConstraint_eq, affineConstraint_eq, affine_rename]
  simp only [Option.map_map, Function.comp_def, renameRow, R1CS.mapRowColumns,
    R1CS.mapCombinationColumns_one, R1CS.mapCombinationColumns_zero]

def direct (expression : Expr) : Option R1CS.Row :=
  (R1CS.directConstraint expression).map R1CS.DirectConstraintResult.row

private theorem direct_witness (output : Nat) (coefficient : F) (expression : Expr) :
    direct (.add (.var output) (.mul (.const coefficient) expression)) =
      if coefficient = -1 then
        (recipe output expression).orElse (fun _ => affineConstraint (.add (.var output) (.mul (.const coefficient) expression)))
      else affineConstraint (.add (.var output) (.mul (.const coefficient) expression)) := by
  by_cases coefficientEquals : coefficient = -1
  · cases found : R1CS.directRecipeRow output expression <;>
      simp [direct, R1CS.directConstraint, coefficientEquals, recipe, found, affineConstraint]
  · simp [direct, R1CS.directConstraint, coefficientEquals, affineConstraint]

theorem direct_rename (column : Nat → Nat) (expression : Expr) :
    direct (renameExpr column expression) = (direct expression).map (renameRow column) := by
  cases expression with
  | var index => exact affineConstraint_rename column (.var index)
  | const value => exact affineConstraint_rename column (.const value)
  | mul left right => exact affineConstraint_rename column (.mul left right)
  | add left right =>
    cases left <;> cases right <;>
      first | exact affineConstraint_rename column _ | skip
    rename_i output factor value
    cases factor <;> first | exact affineConstraint_rename column _ | skip
    rename_i coefficient
    have affineMapped := affineConstraint_rename column
      (.add (.var output) (.mul (.const coefficient) value))
    simp only [renameExpr] at affineMapped
    simp only [renameExpr, direct_witness]
    split_ifs
    · rw [recipe_rename, affineMapped]
      cases recipe output value <;> rfl
    · exact affineMapped

private theorem fresh_eq (expression : Expr) :
    R1CS.constraintFreshCount expression = match direct expression with
      | some _ => 0
      | none => R1CS.mulCount expression := by
  cases found : R1CS.directConstraint expression <;>
    simp [R1CS.constraintFreshCount, direct, found]

theorem constraintFreshCount_rename (column : Nat → Nat) (expression : Expr) :
    R1CS.constraintFreshCount (renameExpr column expression) =
      R1CS.constraintFreshCount expression := by
  rw [fresh_eq, fresh_eq, direct_rename]
  cases direct expression <;> simp [renameExpr_mulCount]

private theorem rows_eq (expression : Expr) (start : Nat) :
    (R1CS.lowerConstraint expression start).rows = match direct expression with
      | some row => [row]
      | none => (R1CS.lowerGenericConstraint expression start).rows := by
  cases found : R1CS.directConstraint expression <;>
    simp [R1CS.lowerConstraint, direct, found]

/-- The optimized compiler preserves every row, not only row satisfaction. -/
theorem lowerConstraint_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (expression : Expr)
    (startBound : inputCount ≤ start) (scope : expression.VarsBelow inputCount) :
    (R1CS.lowerConstraint (renameExpr inputColumn expression) (start + shift)).rows =
      (R1CS.lowerConstraint expression start).rows.map
        (renameRow (relocate inputCount shift inputColumn)) := by
  have sameExpression := renameExpr_congr inputColumn
    (relocate inputCount shift inputColumn) expression scope (by
      intro index below
      exact (relocate_input inputCount shift inputColumn index below).symm)
  have mapped := direct_rename (relocate inputCount shift inputColumn) expression
  rw [← sameExpression] at mapped
  rw [rows_eq, rows_eq, mapped]
  cases found : direct expression with
  | none =>
    exact lowerGenericConstraint_rename inputCount start shift inputColumn
      expression startBound scope
  | some row => rfl

/-- Scratch addresses advance by the same amount for the whole constraint list. -/
theorem lowerConstraints_rename (inputCount start shift : Nat)
    (inputColumn : Nat → Nat) (constraints : List Expr)
    (startBound : inputCount ≤ start)
    (scope : ∀ expression ∈ constraints, expression.VarsBelow inputCount) :
    (R1CS.lowerConstraints (constraints.map (renameExpr inputColumn)) (start + shift)).rows =
      (R1CS.lowerConstraints constraints start).rows.map
        (renameRow (relocate inputCount shift inputColumn)) := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
    have firstScope := scope expression (List.mem_cons_self ..)
    have restScope : ∀ expression ∈ rest, expression.VarsBelow inputCount := by
      intro expression member
      exact scope expression (List.mem_cons_of_mem _ member)
    simp only [List.map_cons, R1CS.lowerConstraints, R1CS.lowerConstraint_next,
      constraintFreshCount_rename, List.map_append]
    rw [lowerConstraint_rename inputCount start shift inputColumn expression startBound firstScope]
    rw [Nat.add_right_comm start shift]
    rw [inductionHypothesis (start + R1CS.constraintFreshCount expression)
      (by omega) restScope]


end NightstreamFPrime.Export.Stage1.ConstraintRenaming
