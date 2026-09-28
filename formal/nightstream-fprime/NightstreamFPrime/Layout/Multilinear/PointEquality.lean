import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.Multilinear.PointEquality

/-!
Owns physical R1CS cost proofs for the reusable owned point-equality gadget.
It counts one equality factor and one extension-field product, then composes
those costs by coordinate-list induction. It does not own a protocol point.
-/

namespace NightstreamFPrime.Layout.Multilinear.PointEquality

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner

namespace Logical

abbrev CoordinateExpr :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.CoordinateExpr

abbrev factorExpr :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.factorExpr

abbrev factorRecipes :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.factorRecipes

abbrev mulRecipes :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.mulRecipes

abbrev materializedAt :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.materializedAt

abbrev compile :=
  NightstreamFPrime.Gadgets.Multilinear.PointEquality.compile

end Logical

/-- Stable physical wire shape for one equality coordinate. -/
structure CoordinateLinear (coordinate : Logical.CoordinateExpr) : Prop where
  left : KExprLinear coordinate.left
  right : KExprLinear coordinate.right

theorem materializedAt_linear (start : Nat) :
    KExprLinear (Logical.materializedAt start) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp [Logical.materializedAt,
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.materializedAt,
      R1CS.mulCount, Nonconstant]

private theorem lowerAffine_add_eq_none_of_left
    (left right : Expr) (leftNone : R1CS.lowerAffine left = none) :
    R1CS.lowerAffine (left + right) = none := by
  simp [R1CS.lowerAffine, leftNone]

private theorem lowerAffine_add_eq_none_of_right
    (left right : Expr) (rightNone : R1CS.lowerAffine right = none) :
    R1CS.lowerAffine (left + right) = none := by
  cases leftResult : R1CS.lowerAffine left <;>
    simp [R1CS.lowerAffine, leftResult, rightNone]

@[simp] private theorem mulCount_sub (left right : Expr) :
    R1CS.mulCount (left - right) =
      R1CS.mulCount left + R1CS.mulCount right + 1 := by
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) right)) = _
  simp [R1CS.mulCount]
  omega

private theorem directConstraint_sub_add_eq_none_of_recipe
    (output : Nat) (first second : Expr)
    (recipeNone : R1CS.lowerAffine (first + second) = none) :
    R1CS.directConstraint (Expr.var output - (first + second)) = none := by
  have directRecipeNone :
      R1CS.directRecipeRow output (first + second) = none := by
    simp [R1CS.directRecipeRow, recipeNone]
  have negativeNone :
      R1CS.lowerAffine ((Expr.const (-1)) * (first + second)) = none := by
    unfold R1CS.lowerAffine
    rw [recipeNone]
  have wholeNone :
      R1CS.lowerAffine (Expr.var output - (first + second)) = none := by
    exact lowerAffine_add_eq_none_of_right _ _ negativeNone
  change R1CS.directConstraint
      (.add (.var output) (.mul (.const (-1)) (.add first second))) = none
  simp only [R1CS.directConstraint]
  split
  · have directRecipeNone' :
        R1CS.directRecipeRow output (Expr.add first second) = none :=
      directRecipeNone
    rw [directRecipeNone']
    unfold R1CS.affineConstraint
    have wholeNone' : R1CS.lowerAffine
        (Expr.add (Expr.var output)
          (Expr.mul (Expr.const (-1)) (Expr.add first second))) = none :=
      wholeNone
    rw [wholeNone']
  · rename_i false
    exact (false trivial).elim

private theorem factorExpr_c0_lowerAffine_eq_none
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.lowerAffine (Logical.factorExpr coordinate).c0 = none := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  let difference := KExpr.sub coordinate.right oneMinusRight
  have differenceNonconstant : Nonconstant difference.c0 := by
    intro value equality
    change Expr.add coordinate.right.c0 _ = Expr.const value at equality
    cases equality
  have firstProductNone :
      R1CS.lowerAffine (coordinate.left.c0 * difference.c0) = none :=
    NightstreamFPrime.Layout.Polynomial.Horner.lowerAffine_mul_eq_none
      linear.left.c0_nonconstant differenceNonconstant
  have productNone :
      R1CS.lowerAffine (KExpr.mul coordinate.left difference).c0 = none := by
    change R1CS.lowerAffine
      (coordinate.left.c0 * difference.c0 +
        7 * coordinate.left.c1 * difference.c1) = none
    exact lowerAffine_add_eq_none_of_left _ _ firstProductNone
  change R1CS.lowerAffine
    (oneMinusRight.c0 + (KExpr.mul coordinate.left difference).c0) = none
  exact lowerAffine_add_eq_none_of_right _ _ productNone

private theorem factorExpr_c1_lowerAffine_eq_none
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.lowerAffine (Logical.factorExpr coordinate).c1 = none := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  let difference := KExpr.sub coordinate.right oneMinusRight
  have differenceNonconstant : Nonconstant difference.c1 := by
    intro value equality
    change Expr.add coordinate.right.c1 _ = Expr.const value at equality
    cases equality
  have firstProductNone :
      R1CS.lowerAffine (coordinate.left.c0 * difference.c1) = none :=
    NightstreamFPrime.Layout.Polynomial.Horner.lowerAffine_mul_eq_none
      linear.left.c0_nonconstant differenceNonconstant
  have productNone :
      R1CS.lowerAffine (KExpr.mul coordinate.left difference).c1 = none := by
    change R1CS.lowerAffine
      (coordinate.left.c0 * difference.c1 +
        coordinate.left.c1 * difference.c0) = none
    exact lowerAffine_add_eq_none_of_left _ _ firstProductNone
  change R1CS.lowerAffine
    (oneMinusRight.c1 + (KExpr.mul coordinate.left difference).c1) = none
  exact lowerAffine_add_eq_none_of_right _ _ productNone

private theorem directConstraint_factor_c0_eq_none
    (output : Nat) (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.directConstraint
      (Expr.var output - (Logical.factorExpr coordinate).c0) = none := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  let difference := KExpr.sub coordinate.right oneMinusRight
  have recipeNone := factorExpr_c0_lowerAffine_eq_none coordinate linear
  change R1CS.directConstraint
    (Expr.var output -
      (oneMinusRight.c0 + (KExpr.mul coordinate.left difference).c0)) = none
  exact directConstraint_sub_add_eq_none_of_recipe _ _ _ recipeNone

private theorem directConstraint_factor_c1_eq_none
    (output : Nat) (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.directConstraint
      (Expr.var output - (Logical.factorExpr coordinate).c1) = none := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  let difference := KExpr.sub coordinate.right oneMinusRight
  have recipeNone := factorExpr_c1_lowerAffine_eq_none coordinate linear
  change R1CS.directConstraint
    (Expr.var output -
      (oneMinusRight.c1 + (KExpr.mul coordinate.left difference).c1)) = none
  exact directConstraint_sub_add_eq_none_of_recipe _ _ _ recipeNone

theorem factorExpr_mulCounts (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.mulCount (Logical.factorExpr coordinate).c0 = 8 ∧
      R1CS.mulCount (Logical.factorExpr coordinate).c1 = 7 := by
  simp [Logical.factorExpr,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.factorExpr,
    KExpr.one, KExpr.sub, KExpr.add, KExpr.mul, R1CS.mulCount,
    mulCount_sub,
    linear.left.c0_mulCount, linear.left.c1_mulCount,
    linear.right.c0_mulCount, linear.right.c1_mulCount]

private theorem factor_c0_freshCount_eq (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.constraintFreshCount
      (Expr.var output - (Logical.factorExpr coordinate).c0) = 9 := by
  unfold R1CS.constraintFreshCount
  rw [directConstraint_factor_c0_eq_none output coordinate linear]
  have counts := factorExpr_mulCounts coordinate linear
  rw [mulCount_sub, counts.1]
  rfl

private theorem factor_c1_freshCount_eq (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.constraintFreshCount
      (Expr.var output - (Logical.factorExpr coordinate).c1) = 8 := by
  unfold R1CS.constraintFreshCount
  rw [directConstraint_factor_c1_eq_none output coordinate linear]
  have counts := factorExpr_mulCounts coordinate linear
  rw [mulCount_sub, counts.2]
  rfl

private theorem factor_c0_rowCount_eq (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.constraintRowCount
      (Expr.var output - (Logical.factorExpr coordinate).c0) = 10 := by
  unfold R1CS.constraintRowCount
  rw [directConstraint_factor_c0_eq_none output coordinate linear]
  have counts := factorExpr_mulCounts coordinate linear
  rw [mulCount_sub, counts.1]
  rfl

private theorem factor_c1_rowCount_eq (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.constraintRowCount
      (Expr.var output - (Logical.factorExpr coordinate).c1) = 9 := by
  unfold R1CS.constraintRowCount
  rw [directConstraint_factor_c1_eq_none output coordinate linear]
  have counts := factorExpr_mulCounts coordinate linear
  rw [mulCount_sub, counts.2]
  rfl

theorem factorRecipes_totalFreshCount (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.totalFreshCount
      (recipeConstraints output (Logical.factorRecipes coordinate)) = 17 := by
  simp only [Logical.factorRecipes,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.factorRecipes,
    recipeConstraints, R1CS.totalFreshCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [factor_c0_freshCount_eq output coordinate linear,
    factor_c1_freshCount_eq (output + 1) coordinate linear]

theorem factorRecipes_totalRowCount (output : Nat)
    (coordinate : Logical.CoordinateExpr)
    (linear : CoordinateLinear coordinate) :
    R1CS.totalRowCount
      (recipeConstraints output (Logical.factorRecipes coordinate)) = 19 := by
  simp only [Logical.factorRecipes,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.factorRecipes,
    recipeConstraints, R1CS.totalRowCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [factor_c0_rowCount_eq output coordinate linear,
    factor_c1_rowCount_eq (output + 1) coordinate linear]

theorem mulRecipes_totalFreshCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left) (rightLinear : KExprLinear right) :
    R1CS.totalFreshCount
      (recipeConstraints output (Logical.mulRecipes left right)) = 7 := by
  simpa [Logical.mulRecipes,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.mulRecipes,
    NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes] using
    NightstreamFPrime.Layout.Polynomial.Horner.mulRecipes_totalFreshCount
      output left right leftLinear rightLinear

theorem mulRecipes_totalRowCount (output : Nat) (left right : KExpr)
    (leftLinear : KExprLinear left) (rightLinear : KExprLinear right) :
    R1CS.totalRowCount
      (recipeConstraints output (Logical.mulRecipes left right)) = 9 := by
  simpa [Logical.mulRecipes,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.mulRecipes,
    NightstreamFPrime.Gadgets.Polynomial.Horner.mulRecipes] using
    NightstreamFPrime.Layout.Polynomial.Horner.mulRecipes_totalRowCount
      output left right leftLinear rightLinear

theorem compile_output_linear_of_nonempty (start : Nat)
    (coordinates : List Logical.CoordinateExpr) (nonempty : coordinates ≠ []) :
    KExprLinear (Logical.compile start coordinates).output := by
  cases coordinates with
  | nil => exact (nonempty rfl).elim
  | cons coordinate rest =>
      cases rest with
      | nil =>
          simpa [Logical.compile,
            NightstreamFPrime.Gadgets.Multilinear.PointEquality.compile] using
            materializedAt_linear start
      | cons next rest =>
          change KExprLinear (Logical.materializedAt
            (start + (Logical.compile start (next :: rest)).recipes.length + 2))
          exact materializedAt_linear _

theorem compile_totalFreshCount (start : Nat)
    (coordinates : List Logical.CoordinateExpr)
    (coordinatesLinear : ∀ coordinate ∈ coordinates,
      CoordinateLinear coordinate) :
    R1CS.totalFreshCount
      (recipeConstraints start (Logical.compile start coordinates).recipes) =
      match coordinates with
      | [] => 0
      | _ => 24 * coordinates.length - 7 := by
  induction coordinates generalizing start with
  | nil => rfl
  | cons coordinate rest inductionHypothesis =>
      cases rest with
      | nil =>
          simpa [Logical.compile,
            NightstreamFPrime.Gadgets.Multilinear.PointEquality.compile] using
            factorRecipes_totalFreshCount start coordinate
              (coordinatesLinear coordinate (by simp))
      | cons next rest =>
          let tail := Logical.compile start (next :: rest)
          let factorStart := start + tail.recipes.length
          let factor := Logical.materializedAt factorStart
          have tailLinear : ∀ current ∈ next :: rest,
              CoordinateLinear current := by
            intro current member
            exact coordinatesLinear current (by simp [member])
          have factorLinear : KExprLinear factor := materializedAt_linear _
          have tailOutputLinear : KExprLinear tail.output :=
            compile_output_linear_of_nonempty start (next :: rest) (by simp)
          rw [show (Logical.compile start (coordinate :: next :: rest)).recipes =
              tail.recipes ++ Logical.factorRecipes coordinate ++
                Logical.mulRecipes factor tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalFreshCount_append,
            recipeConstraints_append, R1CS.totalFreshCount_append,
            inductionHypothesis (start := start) tailLinear,
            factorRecipes_totalFreshCount _ coordinate
              (coordinatesLinear coordinate (by simp)),
            mulRecipes_totalFreshCount _ factor tail.output factorLinear
              tailOutputLinear]
          simp only [List.length_cons]
          omega

theorem compile_totalRowCount (start : Nat)
    (coordinates : List Logical.CoordinateExpr)
    (coordinatesLinear : ∀ coordinate ∈ coordinates,
      CoordinateLinear coordinate) :
    R1CS.totalRowCount
      (recipeConstraints start (Logical.compile start coordinates).recipes) =
      match coordinates with
      | [] => 0
      | _ => 28 * coordinates.length - 9 := by
  induction coordinates generalizing start with
  | nil => rfl
  | cons coordinate rest inductionHypothesis =>
      cases rest with
      | nil =>
          simpa [Logical.compile,
            NightstreamFPrime.Gadgets.Multilinear.PointEquality.compile] using
            factorRecipes_totalRowCount start coordinate
              (coordinatesLinear coordinate (by simp))
      | cons next rest =>
          let tail := Logical.compile start (next :: rest)
          let factorStart := start + tail.recipes.length
          let factor := Logical.materializedAt factorStart
          have tailLinear : ∀ current ∈ next :: rest,
              CoordinateLinear current := by
            intro current member
            exact coordinatesLinear current (by simp [member])
          have factorLinear : KExprLinear factor := materializedAt_linear _
          have tailOutputLinear : KExprLinear tail.output :=
            compile_output_linear_of_nonempty start (next :: rest) (by simp)
          rw [show (Logical.compile start (coordinate :: next :: rest)).recipes =
              tail.recipes ++ Logical.factorRecipes coordinate ++
                Logical.mulRecipes factor tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalRowCount_append,
            recipeConstraints_append, R1CS.totalRowCount_append,
            inductionHypothesis (start := start) tailLinear,
            factorRecipes_totalRowCount _ coordinate
              (coordinatesLinear coordinate (by simp)),
            mulRecipes_totalRowCount _ factor tail.output factorLinear
              tailOutputLinear]
          simp only [List.length_cons]
          omega

theorem compile_totalFreshCount_of_nonempty (start : Nat)
    (coordinates : List Logical.CoordinateExpr)
    (coordinatesLinear : ∀ coordinate ∈ coordinates,
      CoordinateLinear coordinate)
    (nonempty : coordinates ≠ []) :
    R1CS.totalFreshCount
      (recipeConstraints start (Logical.compile start coordinates).recipes) =
      24 * coordinates.length - 7 := by
  cases coordinates with
  | nil => exact (nonempty rfl).elim
  | cons coordinate rest =>
      exact compile_totalFreshCount start (coordinate :: rest)
        coordinatesLinear

theorem compile_totalRowCount_of_nonempty (start : Nat)
    (coordinates : List Logical.CoordinateExpr)
    (coordinatesLinear : ∀ coordinate ∈ coordinates,
      CoordinateLinear coordinate)
    (nonempty : coordinates ≠ []) :
    R1CS.totalRowCount
      (recipeConstraints start (Logical.compile start coordinates).recipes) =
      28 * coordinates.length - 9 := by
  cases coordinates with
  | nil => exact (nonempty rfl).elim
  | cons coordinate rest =>
      exact compile_totalRowCount start (coordinate :: rest)
        coordinatesLinear

private theorem coordinateExprs_linear {variableCount : Nat}
    (interface :
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.Interface
        variableCount)
    (offset : Nat)
    (inputs : ∀ coordinate,
      KExprLinear (interface.left offset coordinate) ∧
        KExprLinear (interface.right offset coordinate)) :
    ∀ coordinate ∈
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs
        interface offset,
      CoordinateLinear coordinate := by
  intro coordinate member
  rw [NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs,
    List.mem_map] at member
  rcases member with ⟨index, _, rfl⟩
  exact ⟨(inputs index).1, (inputs index).2⟩

theorem ownedCircuit_totalFreshCount_of_positive {variableCount : Nat}
    (interface :
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.Interface
        variableCount)
    (offset : Nat) (positive : 0 < variableCount)
    (inputs : ∀ coordinate,
      KExprLinear (interface.left offset coordinate) ∧
        KExprLinear (interface.right offset coordinate)) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.circuit
        interface).main offset)) = 24 * variableCount - 7 := by
  unfold NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.circuit
  rw [NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.main_ops,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.program
  have length :=
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs_length
      interface offset
  have nonempty :
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs
        interface offset ≠ [] := by
    intro empty
    rw [empty] at length
    simp at length
    omega
  rw [compile_totalFreshCount_of_nonempty _ _
    (coordinateExprs_linear interface offset inputs) nonempty, length]

theorem ownedCircuit_totalRowCount_of_positive {variableCount : Nat}
    (interface :
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.Interface
        variableCount)
    (offset : Nat) (positive : 0 < variableCount)
    (inputs : ∀ coordinate,
      KExprLinear (interface.left offset coordinate) ∧
        KExprLinear (interface.right offset coordinate)) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.circuit
        interface).main offset)) = 28 * variableCount - 9 := by
  unfold NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.circuit
  rw [NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.main_ops,
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.flatConstraints_opsAt]
  unfold NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.program
  have length :=
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs_length
      interface offset
  have nonempty :
      NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs
        interface offset ≠ [] := by
    intro empty
    rw [empty] at length
    simp at length
    omega
  rw [compile_totalRowCount_of_nonempty _ _
    (coordinateExprs_linear interface offset inputs) nonempty, length]

end NightstreamFPrime.Layout.Multilinear.PointEquality
