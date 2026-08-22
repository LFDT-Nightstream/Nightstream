import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, `N`.
Obligation: Lower
`N = sum_(i=1)^(K+k) gamma^(i-1) (x_i + 1) x_i (x_i - 1)`
for the fixed strict `b = 2` profile.

Inputs:
- verifier-derived `gamma`;
- 17 source assignments in exact `K + k` order.

Outputs:
- the child-owned strict-norm residual sum.

Constraint groups:
- one symbolic cubic residual per source;
- one reusable Horner multiplication for each of 16 source transitions;
- no expected-output copy row.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.norm_terminal`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Exact syntax shape of one strict-`b = 2` residual or a residual plus a
materialized Horner product. -/
structure ResidualShape (value : KExpr) : Prop where
  c0_mulCount : R1CS.mulCount value.c0 = 10
  c1_mulCount : R1CS.mulCount value.c1 = 9
  c0_nonconstant : Nonconstant value.c0
  c1_nonconstant : Nonconstant value.c1

/-- Stable physical wire shape for the verifier challenge and all source
assignments. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.Interface)
    (offset : Nat) : Prop where
  gamma : KExprLinear (interface.gamma offset)
  sourceAssignment : ∀ source,
    KExprLinear (interface.sourceAssignment offset source)

@[simp] private theorem mulCount_sub (left right : Expr) :
    R1CS.mulCount (left - right) =
      R1CS.mulCount left + R1CS.mulCount right + 1 := by
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) right)) = _
  simp [R1CS.mulCount]
  omega

theorem residualExpr_shape (value : KExpr) (linear : KExprLinear value) :
    ResidualShape
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.residualExpr value) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.residualExpr,
      KExpr.one, KExpr.add, KExpr.sub, KExpr.mul, R1CS.mulCount,
      mulCount_sub,
      linear.c0_mulCount, linear.c1_mulCount]
  · simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.residualExpr,
      KExpr.one, KExpr.add, KExpr.sub, KExpr.mul, R1CS.mulCount,
      mulCount_sub,
      linear.c0_mulCount, linear.c1_mulCount]
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality

private theorem c0_directConstraint_eq_none (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.directConstraint
      (Expr.var output - (KExpr.mul point right).c0) = none := by
  change R1CS.directConstraint
    (Expr.var output -
      (point.c0 * right.c0 + 7 * point.c1 * right.c1)) = none
  apply directConstraint_sub_add_eq_none
  exact lowerAffine_mul_eq_none pointLinear.c0_nonconstant
    rightShape.c0_nonconstant

private theorem c1_directConstraint_eq_none (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.directConstraint
      (Expr.var output - (KExpr.mul point right).c1) = none := by
  change R1CS.directConstraint
    (Expr.var output -
      (point.c0 * right.c1 + point.c1 * right.c0)) = none
  apply directConstraint_sub_add_eq_none
  exact lowerAffine_mul_eq_none pointLinear.c0_nonconstant
    rightShape.c1_nonconstant

private theorem c0_freshCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.constraintFreshCount
      (Expr.var output - (KExpr.mul point right).c0) = 23 := by
  unfold R1CS.constraintFreshCount
  rw [c0_directConstraint_eq_none output point right pointLinear rightShape]
  change R1CS.mulCount
    (.add (.var output)
      (.mul (.const (-1))
        (.add (.mul point.c0 right.c0)
          (.mul (.mul (.const 7) point.c1) right.c1)))) = 23
  simp only [R1CS.mulCount, pointLinear.c0_mulCount,
    pointLinear.c1_mulCount, rightShape.c0_mulCount,
    rightShape.c1_mulCount]

private theorem c1_freshCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.constraintFreshCount
      (Expr.var output - (KExpr.mul point right).c1) = 22 := by
  unfold R1CS.constraintFreshCount
  rw [c1_directConstraint_eq_none output point right pointLinear rightShape]
  change R1CS.mulCount
    (.add (.var output)
      (.mul (.const (-1))
        (.add (.mul point.c0 right.c1)
          (.mul point.c1 right.c0)))) = 22
  simp only [R1CS.mulCount, pointLinear.c0_mulCount,
    pointLinear.c1_mulCount, rightShape.c0_mulCount,
    rightShape.c1_mulCount]

private theorem c0_rowCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.constraintRowCount
      (Expr.var output - (KExpr.mul point right).c0) = 24 := by
  unfold R1CS.constraintRowCount
  rw [c0_directConstraint_eq_none output point right pointLinear rightShape]
  change R1CS.mulCount
      (.add (.var output)
        (.mul (.const (-1))
          (.add (.mul point.c0 right.c0)
            (.mul (.mul (.const 7) point.c1) right.c1)))) + 1 = 24
  simp only [R1CS.mulCount, pointLinear.c0_mulCount,
    pointLinear.c1_mulCount, rightShape.c0_mulCount,
    rightShape.c1_mulCount]

private theorem c1_rowCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.constraintRowCount
      (Expr.var output - (KExpr.mul point right).c1) = 23 := by
  unfold R1CS.constraintRowCount
  rw [c1_directConstraint_eq_none output point right pointLinear rightShape]
  change R1CS.mulCount
      (.add (.var output)
        (.mul (.const (-1))
          (.add (.mul point.c0 right.c1)
            (.mul point.c1 right.c0)))) + 1 = 23
  simp only [R1CS.mulCount, pointLinear.c0_mulCount,
    pointLinear.c1_mulCount, rightShape.c0_mulCount,
    rightShape.c1_mulCount]

theorem mulRecipes_totalFreshCount (output : Nat) (point right : KExpr)
    (pointLinear : KExprLinear point) (rightShape : ResidualShape right) :
    R1CS.totalFreshCount
      (recipeConstraints output (Horner.mulRecipes point right)) = 45 := by
  simp only [Horner.mulRecipes, recipeConstraints, R1CS.totalFreshCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_freshCount_eq output point right pointLinear rightShape,
    c1_freshCount_eq (output + 1) point right pointLinear rightShape]

theorem mulRecipes_totalRowCount (output : Nat) (point right : KExpr)
    (pointLinear : KExprLinear point) (rightShape : ResidualShape right) :
    R1CS.totalRowCount
      (recipeConstraints output (Horner.mulRecipes point right)) = 47 := by
  simp only [Horner.mulRecipes, recipeConstraints, R1CS.totalRowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_rowCount_eq output point right pointLinear rightShape,
    c1_rowCount_eq (output + 1) point right pointLinear rightShape]

private theorem add_product_shape (coefficient : KExpr)
    (shape : ResidualShape coefficient) (start : Nat) :
    ResidualShape (KExpr.add coefficient (Horner.productAt start)) := by
  have productLinear := productAt_linear start
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [KExpr.add, R1CS.mulCount, shape.c0_mulCount,
      productLinear.c0_mulCount]
  · simp [KExpr.add, R1CS.mulCount, shape.c1_mulCount,
      productLinear.c1_mulCount]
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality

theorem compile_output_shape_of_nonempty (start : Nat) (point : KExpr)
    (coefficients : List KExpr) (nonempty : coefficients ≠ [])
    (coefficientsShape : ∀ coefficient ∈ coefficients,
      ResidualShape coefficient) :
    ResidualShape (Horner.compile start point coefficients).output := by
  cases coefficients with
  | nil => exact (nonempty rfl).elim
  | cons coefficient rest =>
      cases rest with
      | nil =>
          simpa [Horner.compile] using
            coefficientsShape coefficient (by simp)
      | cons next rest =>
          let tail := Horner.compile start point (next :: rest)
          change ResidualShape
            (KExpr.add coefficient
              (Horner.productAt (start + tail.recipes.length)))
          exact add_product_shape coefficient
            (coefficientsShape coefficient (by simp)) _

theorem compile_totalFreshCount (start : Nat) (point : KExpr)
    (coefficients : List KExpr) (pointLinear : KExprLinear point)
    (coefficientsShape : ∀ coefficient ∈ coefficients,
      ResidualShape coefficient) :
    R1CS.totalFreshCount
      (recipeConstraints start (Horner.compile start point coefficients).recipes) =
      45 * (coefficients.length - 1) := by
  induction coefficients generalizing start with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil => rfl
      | cons next rest =>
          let tail := Horner.compile start point (next :: rest)
          have tailShape : ∀ current ∈ next :: rest,
              ResidualShape current := by
            intro current member
            exact coefficientsShape current (by simp [member])
          have tailOutputShape : ResidualShape tail.output :=
            compile_output_shape_of_nonempty start point (next :: rest)
              (by simp) tailShape
          rw [show (Horner.compile start point
              (coefficient :: next :: rest)).recipes =
              tail.recipes ++ Horner.mulRecipes point tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalFreshCount_append,
            inductionHypothesis (start := start) tailShape,
            mulRecipes_totalFreshCount _ point tail.output pointLinear
              tailOutputShape]
          simp only [List.length_cons]
          omega

theorem compile_totalRowCount (start : Nat) (point : KExpr)
    (coefficients : List KExpr) (pointLinear : KExprLinear point)
    (coefficientsShape : ∀ coefficient ∈ coefficients,
      ResidualShape coefficient) :
    R1CS.totalRowCount
      (recipeConstraints start (Horner.compile start point coefficients).recipes) =
      47 * (coefficients.length - 1) := by
  induction coefficients generalizing start with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil => rfl
      | cons next rest =>
          let tail := Horner.compile start point (next :: rest)
          have tailShape : ∀ current ∈ next :: rest,
              ResidualShape current := by
            intro current member
            exact coefficientsShape current (by simp [member])
          have tailOutputShape : ResidualShape tail.output :=
            compile_output_shape_of_nonempty start point (next :: rest)
              (by simp) tailShape
          rw [show (Horner.compile start point
              (coefficient :: next :: rest)).recipes =
              tail.recipes ++ Horner.mulRecipes point tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalRowCount_append,
            inductionHypothesis (start := start) tailShape,
            mulRecipes_totalRowCount _ point tail.output pointLinear
              tailOutputShape]
          simp only [List.length_cons]
          omega

private theorem coefficientExprs_shape
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coefficient ∈
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.coefficientExprs
        interface offset,
      ResidualShape coefficient := by
  intro coefficient member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.coefficientExprs,
    List.mem_map] at member
  rcases member with ⟨source, _, rfl⟩
  exact residualExpr_shape _ (inputs.sourceAssignment source)

private theorem flatConstraints_eq_recipeConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.Interface)
    (offset : Nat) :
    flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.circuit interface
        ).main offset) =
      recipeConstraints offset
        (Horner.compile offset (interface.gamma offset)
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.coefficientExprs
            interface offset)).recipes := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.circuit
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.ownedInterface
  rw [Horner.Owned.circuit_ops, Horner.Owned.flatConstraints_opsAt]
  rfl

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.circuit interface
        ).main offset)) = 720 := by
  rw [flatConstraints_eq_recipeConstraints]
  rw [compile_totalFreshCount _ _ _ inputs.gamma
    (coefficientExprs_shape interface offset inputs)]
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.coefficientExprs_length]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.circuit interface
        ).main offset)) = 752 := by
  rw [flatConstraints_eq_recipeConstraints]
  rw [compile_totalRowCount _ _ _ inputs.gamma
    (coefficientExprs_shape interface offset inputs)]
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.coefficientExprs_length]

/-- Exact parent-facing physical footprint for strict base-2 `N`. -/
def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset) :
    R1CS.CircuitFootprint (Formal.normCircuit relation interface) where
  freshColumnCount := fun _ => 720
  physicalRowCount := fun _ => 752
  freshColumnCount_eq := by
    intro offset
    unfold Formal.normCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.normCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.normCircuit relation interface).main offset)) = 720 :=
  (footprint relation interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.normCircuit relation interface).main offset)) = 752 :=
  (footprint relation interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.normCircuit relation interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.normCircuit relation interface).main offset)) = 752 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.normCircuit relation interface).main
        offset) = 32 := by
    unfold Formal.normCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.localLength_eq
        (Formal.normInterface relation interface) offset
  rw [logicalColumns, freshColumnCount_eq relation interface inputs offset]

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal
