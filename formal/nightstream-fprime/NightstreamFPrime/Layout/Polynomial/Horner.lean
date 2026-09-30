import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Gadgets.Polynomial.Horner

/-!
Owns physical lowering for the reusable quadratic-extension Horner compiler.
Each stored product is three rank-one rows and no lowering cell. The cost is
proved structurally from the coefficient-list length.
It does not own any protocol coefficient order or exponent schedule.
-/

namespace NightstreamFPrime.Layout.Polynomial.Horner

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

/-- A symbolic expression is not a literal field constant. -/
def Nonconstant (expression : Expr) : Prop :=
  ∀ value, expression ≠ .const value

/-- The two coordinates contain no multiplication nodes and are not literal
constants. This is the exact syntactic boundary needed for stable Horner
lowering costs. -/
structure KExprLinear (value : KExpr) : Prop where
  c0_mulCount : R1CS.mulCount value.c0 = 0
  c1_mulCount : R1CS.mulCount value.c1 = 0
  c0_nonconstant : Nonconstant value.c0
  c1_nonconstant : Nonconstant value.c1

theorem isAffine_of_mulCount_zero (expression : Expr)
    (count : R1CS.mulCount expression = 0) :
    R1CS.IsAffine expression := by
  induction expression with
  | var index => exact R1CS.isAffine_var index
  | const value => exact R1CS.isAffine_const value
  | add left right leftInduction rightInduction =>
      simp only [R1CS.mulCount, Nat.add_eq_zero_iff] at count
      exact R1CS.IsAffine.add (leftInduction count.1) (rightInduction count.2)
  | mul left right leftInduction rightInduction =>
      simp [R1CS.mulCount] at count

theorem KExprLinear.isAffine {value : KExpr}
    (linear : KExprLinear value) :
    R1CS.IsAffine value.c0 ∧ R1CS.IsAffine value.c1 :=
  ⟨isAffine_of_mulCount_zero value.c0 linear.c0_mulCount,
    isAffine_of_mulCount_zero value.c1 linear.c1_mulCount⟩

theorem KExprLinear.add {left right : KExpr}
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    KExprLinear (KExpr.add left right) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [KExpr.add, R1CS.mulCount, leftLinear.c0_mulCount,
      rightLinear.c0_mulCount]
  · simp [KExpr.add, R1CS.mulCount, leftLinear.c1_mulCount,
      rightLinear.c1_mulCount]
  · simp [KExpr.add, Nonconstant]
  · simp [KExpr.add, Nonconstant]

theorem productAt_linear (start : Nat) :
    KExprLinear
      (NightstreamFPrime.Gadgets.Polynomial.Horner.productAt start) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [NightstreamFPrime.Gadgets.Polynomial.Horner.productAt,
      R1CS.mulCount, Nonconstant]

theorem lowerAffine_mul_eq_none {left right : Expr}
    (leftNonconstant : Nonconstant left)
    (rightNonconstant : Nonconstant right) :
    R1CS.lowerAffine (left * right) = none := by
  cases left <;> cases right <;>
    simp_all [Nonconstant, R1CS.lowerAffine]

/-- A recipe that is a sum of two non-affine parts has no one-row lowering. -/
theorem directConstraint_sub_add_eq_none
    (output : Nat) (left right : Expr)
    (leftNone : R1CS.lowerAffine left = none)
    (rightNone : R1CS.lowerAffine right = none) :
    R1CS.directConstraint (Expr.var output - (left + right)) = none := by
  change R1CS.directConstraint
    (.add (.var output) (.mul (.const (-1)) (.add left right))) = none
  cases left <;>
    simp_all [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.productSumRecipeRow?, R1CS.affineConstraint, R1CS.lowerAffine]

private theorem scaledHigh_lowerAffine_none (left right : KExpr)
    (rightLinear : KExprLinear right) :
    R1CS.lowerAffine ((7 : Expr) * left.c1 * right.c1) = none :=
  lowerAffine_mul_eq_none (by simp [Nonconstant, HMul.hMul, Mul.mul])
    rightLinear.c1_nonconstant

private theorem c0_directConstraint_eq_none (output : Nat)
    (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.directConstraint
      (Expr.var output - (KExpr.mul left right).c0) = none := by
  change R1CS.directConstraint
    (Expr.var output -
      (left.c0 * right.c0 + (7 : Expr) * left.c1 * right.c1)) = none
  apply directConstraint_sub_add_eq_none
  · exact lowerAffine_mul_eq_none leftLinear.c0_nonconstant
      rightLinear.c0_nonconstant
  · exact scaledHigh_lowerAffine_none left right rightLinear

private theorem c1_directConstraint_eq_none (output : Nat)
    (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.directConstraint
      (Expr.var output - (KExpr.mul left right).c1) = none := by
  change R1CS.directConstraint
    (Expr.var output -
      (left.c0 * right.c1 + left.c1 * right.c0)) = none
  apply directConstraint_sub_add_eq_none
  · exact lowerAffine_mul_eq_none leftLinear.c0_nonconstant
      rightLinear.c1_nonconstant
  · exact lowerAffine_mul_eq_none leftLinear.c1_nonconstant
      rightLinear.c0_nonconstant

private theorem c0_constraint_mulCount_eq (output : Nat)
    (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.mulCount
      (Expr.var output - (KExpr.mul left right).c0) = 4 := by
  change R1CS.mulCount
    (.add (.var output)
      (.mul (.const (-1))
        (.add (.mul left.c0 right.c0)
          (.mul (.mul (.const 7) left.c1) right.c1)))) = 4
  simp only [R1CS.mulCount, leftLinear.c0_mulCount,
    leftLinear.c1_mulCount, rightLinear.c0_mulCount,
    rightLinear.c1_mulCount]

private theorem c1_constraint_mulCount_eq (output : Nat)
    (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.mulCount
      (Expr.var output - (KExpr.mul left right).c1) = 3 := by
  change R1CS.mulCount
    (.add (.var output)
      (.mul (.const (-1))
        (.add (.mul left.c0 right.c1)
          (.mul left.c1 right.c0)))) = 3
  simp only [R1CS.mulCount, leftLinear.c0_mulCount,
    leftLinear.c1_mulCount, rightLinear.c0_mulCount,
    rightLinear.c1_mulCount]

private theorem c0_freshCount_eq (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.constraintFreshCount
      (Expr.var output - (KExpr.mul left right).c0) = 4 := by
  unfold R1CS.constraintFreshCount
  rw [c0_directConstraint_eq_none output left right leftLinear rightLinear]
  exact c0_constraint_mulCount_eq output left right leftLinear rightLinear

private theorem c1_freshCount_eq (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.constraintFreshCount
      (Expr.var output - (KExpr.mul left right).c1) = 3 := by
  unfold R1CS.constraintFreshCount
  rw [c1_directConstraint_eq_none output left right leftLinear rightLinear]
  exact c1_constraint_mulCount_eq output left right leftLinear rightLinear

private theorem c0_rowCount_eq (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.constraintRowCount
      (Expr.var output - (KExpr.mul left right).c0) = 5 := by
  unfold R1CS.constraintRowCount
  rw [c0_directConstraint_eq_none output left right leftLinear rightLinear]
  rw [c0_constraint_mulCount_eq output left right leftLinear rightLinear]

private theorem c1_rowCount_eq (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.constraintRowCount
      (Expr.var output - (KExpr.mul left right).c1) = 4 := by
  unfold R1CS.constraintRowCount
  rw [c1_directConstraint_eq_none output left right leftLinear rightLinear]
  rw [c1_constraint_mulCount_eq output left right leftLinear rightLinear]

/-- Cost of the two-cell product whose recipes are the nested components of
`KExpr.mul`. Only gadgets that still store a product in this form use it. -/
theorem nestedMulRecipes_totalFreshCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.totalFreshCount
      (recipeConstraints output
        [(KExpr.mul left right).c0, (KExpr.mul left right).c1]) = 7 := by
  simp only [recipeConstraints, R1CS.totalFreshCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_freshCount_eq output left right leftLinear rightLinear,
    c1_freshCount_eq (output + 1) left right leftLinear rightLinear]

theorem nestedMulRecipes_totalRowCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.totalRowCount
      (recipeConstraints output
        [(KExpr.mul left right).c0, (KExpr.mul left right).c1]) = 9 := by
  simp only [recipeConstraints, R1CS.totalRowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_rowCount_eq output left right leftLinear rightLinear,
    c1_rowCount_eq (output + 1) left right leftLinear rightLinear]

/-- Each of the three stored-product recipes is one rank-one row when both
operands are affine. -/
theorem mulRecipes_direct (output : Nat) (left right : KExpr)
    (leftAffine : R1CS.IsAffine left.c0 ∧ R1CS.IsAffine left.c1)
    (rightAffine : R1CS.IsAffine right.c0 ∧ R1CS.IsAffine right.c1) :
    R1CS.RecipesDirect output
      (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes output
        left right) := by
  refine ⟨R1CS.IsDirectRecipe.mul output leftAffine.2 rightAffine.2,
    R1CS.IsDirectRecipe.mul_add (output + 1) leftAffine.1 rightAffine.1
      (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var output)),
    R1CS.IsDirectRecipe.mul_add (output + 1 + 1)
      (R1CS.IsAffine.add leftAffine.1 leftAffine.2)
      (R1CS.IsAffine.add rightAffine.1 rightAffine.2)
      (R1CS.IsAffine.add
        (R1CS.IsAffine.const_mul (-1) (R1CS.isAffine_var (output + 1)))
        (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var output))),
    trivial⟩

/-- Both coordinates are affine expressions. -/
def KAffine (value : KExpr) : Prop :=
  R1CS.IsAffine value.c0 ∧ R1CS.IsAffine value.c1

theorem KExprLinear.kAffine {value : KExpr} (linear : KExprLinear value) :
    KAffine value :=
  linear.isAffine

/-- The accumulator of an affine coefficient list stays affine. -/
theorem compile_output_kAffine (start : Nat) (point : KExpr) :
    ∀ coefficients : List KExpr,
      (∀ coefficient ∈ coefficients, KAffine coefficient) →
      KAffine
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          coefficients).output
  | [], _ => ⟨R1CS.isAffine_const _, R1CS.isAffine_const _⟩
  | [coefficient], coefficientsAffine => coefficientsAffine coefficient (by simp)
  | coefficient :: _ :: _, coefficientsAffine =>
      ⟨R1CS.IsAffine.add (coefficientsAffine coefficient (by simp)).1
          (R1CS.isAffine_var _),
        R1CS.IsAffine.add (coefficientsAffine coefficient (by simp)).2
          (R1CS.isAffine_var _)⟩

/-- Every Horner recipe over affine operands is one rank-one row. -/
theorem compile_recipesDirect (start : Nat) (point : KExpr)
    (coefficients : List KExpr) (pointAffine : KAffine point)
    (coefficientsAffine : ∀ coefficient ∈ coefficients,
      KAffine coefficient) :
    R1CS.RecipesDirect start
      (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        coefficients).recipes := by
  induction coefficients generalizing start with
  | nil => trivial
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil => trivial
      | cons next rest =>
          let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile
            start point (next :: rest)
          have tailAffine : ∀ current ∈ next :: rest, KAffine current :=
            fun current member =>
              coefficientsAffine current (by simp [member])
          change R1CS.RecipesDirect start
            (tail.recipes ++
              NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes
                (start + tail.recipes.length) point tail.output)
          exact R1CS.recipesDirect_append start _ _
            (inductionHypothesis (start := start) tailAffine)
            (mulRecipes_direct _ point tail.output pointAffine
              (compile_output_kAffine start point (next :: rest) tailAffine))

theorem mulRecipes_totalFreshCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.totalFreshCount
      (recipeConstraints output
        (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes output
          left right)) = 0 :=
  R1CS.recipeConstraints_totalFreshCount output _
    (mulRecipes_direct output left right leftLinear.isAffine
      rightLinear.isAffine)

theorem mulRecipes_totalRowCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left)
    (rightLinear : KExprLinear right) :
    R1CS.totalRowCount
      (recipeConstraints output
        (NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes output
          left right)) = 3 :=
  R1CS.recipeConstraints_totalRowCount output _
    (mulRecipes_direct output left right leftLinear.isAffine
      rightLinear.isAffine)

theorem compile_output_linear (start : Nat) (point : KExpr)
    (coefficients : List KExpr)
    (coefficientsNonempty : coefficients ≠ [])
    (coefficientsLinear : ∀ coefficient ∈ coefficients,
      KExprLinear coefficient) :
    KExprLinear
      (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
        coefficients).output := by
  induction coefficients generalizing start with
  | nil => simp at coefficientsNonempty
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil =>
          simpa [NightstreamFPrime.Gadgets.Polynomial.Horner.compile] using
            coefficientsLinear coefficient (by simp)
      | cons next rest =>
          let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile
            start point (next :: rest)
          have tailCoefficientsLinear : ∀ current ∈ next :: rest,
              KExprLinear current := by
            intro current member
            exact coefficientsLinear current (by simp [member])
          have tailLinear : KExprLinear tail.output := by
            exact inductionHypothesis (start := start) (by simp)
              tailCoefficientsLinear
          have coefficientLinear : KExprLinear coefficient :=
            coefficientsLinear coefficient (by simp)
          have productLinear : KExprLinear
              (NightstreamFPrime.Gadgets.Polynomial.Horner.productAt
                (start + tail.recipes.length)) :=
            productAt_linear _
          simpa [NightstreamFPrime.Gadgets.Polynomial.Horner.compile, tail] using
            KExprLinear.add coefficientLinear productLinear

theorem compile_totalFreshCount (start : Nat) (point : KExpr)
    (coefficients : List KExpr)
    (pointLinear : KExprLinear point)
    (coefficientsLinear : ∀ coefficient ∈ coefficients,
      KExprLinear coefficient) :
    R1CS.totalFreshCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          coefficients).recipes) = 0 := by
  induction coefficients generalizing start with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil => rfl
      | cons next rest =>
          let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile
            start point (next :: rest)
          have tailCoefficientsLinear : ∀ current ∈ next :: rest,
              KExprLinear current := by
            intro current member
            exact coefficientsLinear current (by simp [member])
          have tailOutputLinear : KExprLinear tail.output :=
            compile_output_linear start point (next :: rest) (by simp)
              tailCoefficientsLinear
          rw [show (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
              (coefficient :: next :: rest)).recipes =
              tail.recipes ++
                NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes
                  (start + tail.recipes.length) point tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalFreshCount_append,
            inductionHypothesis (start := start) tailCoefficientsLinear,
            mulRecipes_totalFreshCount _ point tail.output pointLinear
              tailOutputLinear]

theorem compile_totalRowCount (start : Nat) (point : KExpr)
    (coefficients : List KExpr)
    (pointLinear : KExprLinear point)
    (coefficientsLinear : ∀ coefficient ∈ coefficients,
      KExprLinear coefficient) :
    R1CS.totalRowCount
      (recipeConstraints start
        (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
          coefficients).recipes) =
      3 * (coefficients.length - 1) := by
  induction coefficients generalizing start with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil => rfl
      | cons next rest =>
          let tail := NightstreamFPrime.Gadgets.Polynomial.Horner.compile
            start point (next :: rest)
          have tailCoefficientsLinear : ∀ current ∈ next :: rest,
              KExprLinear current := by
            intro current member
            exact coefficientsLinear current (by simp [member])
          have tailOutputLinear : KExprLinear tail.output :=
            compile_output_linear start point (next :: rest) (by simp)
              tailCoefficientsLinear
          rw [show (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start point
              (coefficient :: next :: rest)).recipes =
              tail.recipes ++
                NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes
                  (start + tail.recipes.length) point tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalRowCount_append,
            inductionHypothesis (start := start) tailCoefficientsLinear,
            mulRecipes_totalRowCount _ point tail.output pointLinear
              tailOutputLinear]
          simp only [List.length_cons]
          omega

theorem ownedCircuit_totalFreshCount
    (interface :
      NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.Interface)
    (offset : Nat) (pointLinear : KExprLinear (interface.point offset))
    (coefficientsLinear : ∀ coefficient ∈ interface.coefficients offset,
      KExprLinear coefficient) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit interface
        ).main offset)) = 0 := by
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact compile_totalFreshCount offset (interface.point offset)
    (interface.coefficients offset) pointLinear coefficientsLinear

theorem ownedCircuit_totalRowCount
    (interface :
      NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.Interface)
    (offset : Nat) (pointLinear : KExprLinear (interface.point offset))
    (coefficientsLinear : ∀ coefficient ∈ interface.coefficients offset,
      KExprLinear coefficient) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit interface
        ).main offset)) =
      3 * ((interface.coefficients offset).length - 1) := by
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  exact compile_totalRowCount offset (interface.point offset)
    (interface.coefficients offset) pointLinear coefficientsLinear

end NightstreamFPrime.Layout.Polynomial.Horner
