import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness

/-!
Paper authority: SuperNeo v1.2, section 7.3, Step 4, `N`.
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
- `Formal.opsAt`, child `piccs.v1_2.norm_terminal`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.NormTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
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
  c0_nonAffine : R1CS.lowerAffine value.c0 = none
  c1_nonAffine : R1CS.lowerAffine value.c1 = none

/-- Stable physical wire shape for the verifier challenge and all source
assignments. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.Interface)
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
      (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.residualExpr value) := by
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl⟩
  · simp [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.residualExpr,
      KExpr.one, KExpr.add, KExpr.sub, KExpr.mul, R1CS.mulCount,
      mulCount_sub,
      linear.c0_mulCount, linear.c1_mulCount]
  · simp [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.residualExpr,
      KExpr.one, KExpr.add, KExpr.sub, KExpr.mul, R1CS.mulCount,
      mulCount_sub,
      linear.c0_mulCount, linear.c1_mulCount]
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality
  · intro constant equality
    change Expr.add _ _ = Expr.const constant at equality
    cases equality

/-- The exact strict-base-2 residual uses no variable outside its source
assignment's range. -/
theorem residualExpr_varsBelow (value : KExpr) (bound : Nat)
    (below : value.VarsBelow bound) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.residualExpr
      value).VarsBelow bound := by
  have oneBelow : KExpr.one.VarsBelow bound := by
    simp [KExpr.one, KExpr.VarsBelow, Expr.VarsBelow]
  have plusBelow := KExpr.add_varsBelow value KExpr.one bound below oneBelow
  have firstProductBelow := KExpr.mul_varsBelow
    (KExpr.add value KExpr.one) value bound plusBelow below
  have minusBelow : (KExpr.sub value KExpr.one).VarsBelow bound := by
    unfold KExpr.sub KExpr.VarsBelow
    exact ⟨Expr.VarsBelow.sub _ _ bound below.1 oneBelow.1,
      Expr.VarsBelow.sub _ _ bound below.2 oneBelow.2⟩
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.residualExpr
  exact KExpr.mul_varsBelow _ _ bound firstProductBelow minusBelow

/-- The owned norm result lies below the canonical final-identity child
start. -/
theorem output_varsBelow_finalIdentity
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) (env : Env)
    (assumptions :
      (Formal.normCircuit relation (Formal.atOffset interface parentOffset)
        ).assumptions (Formal.normOffset relation interface parentOffset) env) :
    (Formal.normOutput relation (Formal.atOffset interface parentOffset)
      (Formal.finalIdentityOffset relation interface parentOffset)).VarsBelow
        (Formal.finalIdentityOffset relation interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  have childAssumptions : NormTerminal.Assumptions
      (Formal.normInterface relation frozen) (Formal.normStart frozen) env := by
    rw [Formal.normStart_atOffset relation interface parentOffset]
    exact assumptions
  have below := NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output_varsBelow
    (NormTerminal.ownedInterface (Formal.normInterface relation frozen))
    (Formal.normStart frozen) env childAssumptions
  have outputEq : Formal.normOutput relation frozen
      (Formal.finalIdentityOffset relation interface parentOffset) =
      NormTerminal.output (Formal.normInterface relation frozen)
        (Formal.normStart frozen) := by
    rfl
  have boundEq : Formal.finalIdentityOffset relation interface parentOffset =
      Formal.normStart frozen + localLength (Circuit.ops
        (NormTerminal.circuit (Formal.normInterface relation frozen)).main
        (Formal.normStart frozen)) := by
    unfold Formal.finalIdentityOffset Formal.nextOffset Formal.childLength
    rw [Formal.normStart_atOffset relation interface parentOffset]
    unfold Formal.normCircuit
    rw [FormalCircuit.withConstantFootprint_main]
  rw [outputEq, boundEq]
  exact below

/-! The norm residual operand is not affine, so its three product recipes use
the general lowering. -/

private theorem high_directConstraint_eq_none (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.directConstraint
      (Expr.var output - point.c1 * right.c1) = none := by
  have productNone : R1CS.lowerAffine (.mul point.c1 right.c1) = none :=
    lowerAffine_mul_eq_none pointLinear.c1_nonconstant
      rightShape.c1_nonconstant
  change R1CS.directConstraint
    (.add (.var output) (.mul (.const (-1)) (.mul point.c1 right.c1))) = none
  cases pointAffine : R1CS.lowerAffine point.c1 <;>
    simp [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.affineConstraint, R1CS.lowerAffine, productNone,
      rightShape.c1_nonAffine, pointAffine]

private theorem c0_directConstraint_eq_none (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.directConstraint
      (Expr.var (output + 1) -
        (point.c0 * right.c0 + 7 * Expr.var output)) = none := by
  have productNone : R1CS.lowerAffine (.mul point.c0 right.c0) = none :=
    lowerAffine_mul_eq_none pointLinear.c0_nonconstant
      rightShape.c0_nonconstant
  change R1CS.directConstraint
    (.add (.var (output + 1))
      (.mul (.const (-1))
        (.add (.mul point.c0 right.c0)
          (.mul (.const 7) (.var output))))) = none
  cases pointAffine : R1CS.lowerAffine point.c0 <;>
    simp [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.productSumRecipeRow?, R1CS.affineConstraint, R1CS.lowerAffine,
      productNone, rightShape.c0_nonAffine, pointAffine]

private theorem c1_directConstraint_eq_none (output : Nat)
    (point right : KExpr) (rightShape : ResidualShape right) :
    R1CS.directConstraint
      (Expr.var (output + 1 + 1) -
        ((point.c0 + point.c1) * (right.c0 + right.c1) +
          (-Expr.var (output + 1) + 6 * Expr.var output))) = none := by
  change R1CS.directConstraint
    (.add (.var (output + 1 + 1))
      (.mul (.const (-1))
        (.add (.mul (.add point.c0 point.c1) (.add right.c0 right.c1))
          (.add (.mul (.const (-1)) (.var (output + 1)))
            (.mul (.const 6) (.var output)))))) = none
  cases pointAffine : R1CS.lowerAffine (.add point.c0 point.c1) <;>
    simp_all [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.productSumRecipeRow?, R1CS.affineConstraint, R1CS.lowerAffine,
      rightShape.c0_nonAffine]

private theorem high_mulCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.mulCount (Expr.var output - point.c1 * right.c1) = 11 := by
  change R1CS.mulCount
    (.add (.var output) (.mul (.const (-1)) (.mul point.c1 right.c1))) = 11
  simp only [R1CS.mulCount, pointLinear.c1_mulCount, rightShape.c1_mulCount]

private theorem c0_mulCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.mulCount
      (Expr.var (output + 1) -
        (point.c0 * right.c0 + 7 * Expr.var output)) = 13 := by
  change R1CS.mulCount
    (.add (.var (output + 1))
      (.mul (.const (-1))
        (.add (.mul point.c0 right.c0)
          (.mul (.const 7) (.var output))))) = 13
  simp only [R1CS.mulCount, pointLinear.c0_mulCount, rightShape.c0_mulCount]

private theorem c1_mulCount_eq (output : Nat)
    (point right : KExpr) (pointLinear : KExprLinear point)
    (rightShape : ResidualShape right) :
    R1CS.mulCount
      (Expr.var (output + 1 + 1) -
        ((point.c0 + point.c1) * (right.c0 + right.c1) +
          (-Expr.var (output + 1) + 6 * Expr.var output))) = 23 := by
  change R1CS.mulCount
    (.add (.var (output + 1 + 1))
      (.mul (.const (-1))
        (.add (.mul (.add point.c0 point.c1) (.add right.c0 right.c1))
          (.add (.mul (.const (-1)) (.var (output + 1)))
            (.mul (.const 6) (.var output)))))) = 23
  simp only [R1CS.mulCount, pointLinear.c0_mulCount, pointLinear.c1_mulCount,
    rightShape.c0_mulCount, rightShape.c1_mulCount]

theorem mulRecipes_totalFreshCount (output : Nat) (point right : KExpr)
    (pointLinear : KExprLinear point) (rightShape : ResidualShape right) :
    R1CS.totalFreshCount
      (recipeConstraints output (Horner.mulRecipes output point right)) =
      47 := by
  simp only [Horner.mulRecipes, recipeConstraints, R1CS.totalFreshCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    R1CS.constraintFreshCount]
  rw [high_directConstraint_eq_none output point right pointLinear rightShape,
    c0_directConstraint_eq_none output point right pointLinear rightShape,
    c1_directConstraint_eq_none output point right rightShape,
    high_mulCount_eq output point right pointLinear rightShape,
    c0_mulCount_eq output point right pointLinear rightShape,
    c1_mulCount_eq output point right pointLinear rightShape]
  rfl

theorem mulRecipes_totalRowCount (output : Nat) (point right : KExpr)
    (pointLinear : KExprLinear point) (rightShape : ResidualShape right) :
    R1CS.totalRowCount
      (recipeConstraints output (Horner.mulRecipes output point right)) =
      50 := by
  simp only [Horner.mulRecipes, recipeConstraints, R1CS.totalRowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    R1CS.constraintRowCount]
  rw [high_directConstraint_eq_none output point right pointLinear rightShape,
    c0_directConstraint_eq_none output point right pointLinear rightShape,
    c1_directConstraint_eq_none output point right rightShape,
    high_mulCount_eq output point right pointLinear rightShape,
    c0_mulCount_eq output point right pointLinear rightShape,
    c1_mulCount_eq output point right pointLinear rightShape]
  rfl

private theorem add_product_shape (coefficient : KExpr)
    (shape : ResidualShape coefficient) (start : Nat) :
    ResidualShape (KExpr.add coefficient (Horner.productAt start)) := by
  have productLinear := productAt_linear start
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · change R1CS.lowerAffine (.add coefficient.c0 _) = none
    simp [R1CS.lowerAffine, shape.c0_nonAffine]
  · change R1CS.lowerAffine (.add coefficient.c1 _) = none
    simp [R1CS.lowerAffine, shape.c1_nonAffine]

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
              tail.recipes ++
                Horner.mulRecipes (start + tail.recipes.length) point
                  tail.output by rfl]
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
      50 * (coefficients.length - 1) := by
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
              tail.recipes ++
                Horner.mulRecipes (start + tail.recipes.length) point
                  tail.output by rfl]
          rw [recipeConstraints_append, R1CS.totalRowCount_append,
            inductionHypothesis (start := start) tailShape,
            mulRecipes_totalRowCount _ point tail.output pointLinear
              tailOutputShape]
          simp only [List.length_cons]
          omega

private theorem coefficientExprs_shape
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coefficient ∈
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs
        interface offset,
      ResidualShape coefficient := by
  intro coefficient member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs,
    List.mem_map] at member
  rcases member with ⟨source, _, rfl⟩
  exact residualExpr_shape _ (inputs.sourceAssignment source)

private theorem flatConstraints_eq_recipeConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.Interface)
    (offset : Nat) :
    flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.circuit interface
        ).main offset) =
      recipeConstraints offset
        (Horner.compile offset (interface.gamma offset)
          (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs
            interface offset)).recipes := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.circuit
    NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.ownedInterface
  rw [Horner.Owned.circuit_ops, Horner.Owned.flatConstraints_opsAt]
  rfl

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.circuit interface
        ).main offset)) = 752 := by
  rw [flatConstraints_eq_recipeConstraints]
  rw [compile_totalFreshCount _ _ _ inputs.gamma
    (coefficientExprs_shape interface offset inputs)]
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs_length]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.circuit interface
        ).main offset)) = 800 := by
  rw [flatConstraints_eq_recipeConstraints]
  rw [compile_totalRowCount _ _ _ inputs.gamma
    (coefficientExprs_shape interface offset inputs)]
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs_length]

/-- Exact parent-facing physical footprint for strict base-2 `N`. -/
def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset) :
    R1CS.CircuitFootprint (Formal.normCircuit relation interface) where
  freshColumnCount := fun _ => 752
  physicalRowCount := fun _ => 800
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
      (Formal.normCircuit relation interface).main offset)) = 752 :=
  (footprint relation interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.normInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.normCircuit relation interface).main offset)) = 800 :=
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
        (Formal.normCircuit relation interface).main offset)) = 800 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.normCircuit relation interface).main
        offset) = 48 := by
    unfold Formal.normCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.localLength_eq
        (Formal.normInterface relation interface) offset
  rw [logicalColumns, freshColumnCount_eq relation interface inputs offset]

/-- Exact residual shape exported to the final identity. -/
theorem output_shape
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.normInterface relation interface)
      (Formal.normStart interface)) :
    ResidualShape (Formal.normOutput relation interface offset) := by
  unfold Formal.normOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
  apply compile_output_shape_of_nonempty
  · intro empty
    have coefficientExprsEmpty :
        NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs
          (Formal.normInterface relation interface) (Formal.normStart interface) =
          [] := by
      simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.ownedInterface]
        using empty
    have length :=
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.coefficientExprs_length
        (Formal.normInterface relation interface) (Formal.normStart interface)
    rw [coefficientExprsEmpty] at length
    simp at length
  · exact coefficientExprs_shape (Formal.normInterface relation interface)
      (Formal.normStart interface) inputs

end NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.NormTerminal
