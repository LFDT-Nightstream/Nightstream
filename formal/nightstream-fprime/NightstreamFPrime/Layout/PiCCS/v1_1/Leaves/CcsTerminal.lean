import NightstreamFPrime.Layout.Polynomial.Sparse
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, `F`.
Obligation: Lower the materialized evaluation of the fixed 74-term selective
constraint polynomial over all 14 `Eval_A` matrix images.

Inputs:
- 13 meaningful selective matrix images;
- one canonical-zero matrix image in slot 13;
- the relation-owned sparse polynomial. Pad and `Eval_K` do not enter.

Outputs:
- two child-owned residual wires consumed by final identity.

Constraint groups:
- C1: one checked sparse-evaluation recipe for the real component;
- C2: one checked sparse-evaluation recipe for the extension component.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.ccs_terminal`.

The structural sparse cost model proves 20,792 fresh lowering columns and
20,794 physical rows. No proof evaluates emitted circuit data.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Interface)
    (offset : Nat) : Prop where
  freshMatrix : ∀ matrix,
    KExprLinear (interface.freshMatrix offset matrix)

/-- The two child-owned CCS residual wires lie below the canonical norm child
start. -/
theorem output_varsBelow_norm
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) :
    (Formal.ccsOutput relation (Formal.atOffset interface parentOffset)
      (Formal.normOffset relation interface parentOffset)).VarsBelow
        (Formal.normOffset relation interface parentOffset) := by
  have normOffsetEq : Formal.normOffset relation interface parentOffset =
      Formal.ccsOffset interface parentOffset + 2 := by
    calc
      Formal.normOffset relation interface parentOffset =
          Formal.normStart (Formal.atOffset interface parentOffset) :=
        (Formal.normStart_atOffset relation interface parentOffset).symm
      _ = _ := by
        unfold Formal.normStart
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount
        rw [Formal.ccsStart_atOffset interface parentOffset]
  unfold Formal.ccsOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.output
    NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.output
    KExpr.VarsBelow Expr.VarsBelow
  rw [Formal.ccsStart_atOffset interface parentOffset, normOffsetEq]
  omega

private def expression
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : KExpr :=
  NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.polynomial relation)
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.sparseInterface
      (Formal.ccsInterface relation interface)) offset

private theorem linearPolynomialCounts_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.Polynomial.Sparse.linearPolynomialCounts
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.polynomial
          relation) =
      ⟨10432, 10358⟩ := by
  change NightstreamFPrime.Layout.Polynomial.Sparse.linearPolynomialCounts
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        ProductionRelation.polynomial) = _
  rw [NightstreamFPrime.Layout.Polynomial.Sparse.linearPolynomialCounts_liftConstraintPolynomial]
  rfl

theorem expression_mulCounts
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.mulCount (expression relation interface offset).c0 = 10432 ∧
      R1CS.mulCount (expression relation interface offset).c1 = 10358 := by
  have counts :=
    NightstreamFPrime.Layout.Polynomial.Sparse.expressionCounts_evaluate_of_linear
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.polynomial
          relation)
        (fun matrix =>
          (Formal.ccsInterface relation interface).freshMatrix offset matrix)
        inputs.freshMatrix
  rw [linearPolynomialCounts_eq relation] at counts
  constructor
  · have c0 := congrArg
      NightstreamFPrime.Layout.Polynomial.Sparse.Counts.c0 counts
    simpa [expression,
      NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.sparseInterface,
      NightstreamFPrime.Layout.Polynomial.Sparse.expressionCounts] using c0
  · have c1 := congrArg
      NightstreamFPrime.Layout.Polynomial.Sparse.Counts.c1 counts
    simpa [expression,
      NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.sparseInterface,
      NightstreamFPrime.Layout.Polynomial.Sparse.expressionCounts] using c1

private def bitTerm : Monomial K productionShape.matrixCount :=
  ConstraintPolynomialLift.liftMonomial K.embed
    (ProductionRelation.SelectivePolynomial.monomial 1
      (ProductionRelation.SelectivePolynomial.powers
        (bit := 2) (generalSelector := 1)))

private theorem bitTerm_mem
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    bitTerm ∈
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.polynomial
        relation).terms := by
  change bitTerm ∈
    (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      ProductionRelation.polynomial).terms
  change ConstraintPolynomialLift.liftMonomial K.embed
      (ProductionRelation.SelectivePolynomial.monomial 1
        (ProductionRelation.SelectivePolynomial.powers
          (bit := 2) (generalSelector := 1))) ∈
    ProductionRelation.polynomial.terms.map
      (ConstraintPolynomialLift.liftMonomial K.embed)
  apply List.mem_map_of_mem
  change _ ∈ ProductionRelation.SelectivePolynomial.terms
  unfold ProductionRelation.SelectivePolynomial.terms
  apply List.mem_append_left
  exact List.mem_cons_self

private def bitIndex : Fin productionShape.matrixCount :=
  ⟨0, by
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape]⟩

private def generalSelectorIndex : Fin productionShape.matrixCount :=
  ⟨1, by
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape]⟩

private theorem bitTerm_evaluate_eq
    (point : Fin productionShape.matrixCount → KExpr) :
    NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial bitTerm point =
      KExpr.mul
        (KExpr.mul
          (NightstreamFPrime.Gadgets.Polynomial.Sparse.constant (K.embed 1))
          (KExpr.mul
            (KExpr.mul KExpr.one (point bitIndex))
            (point bitIndex)))
        (KExpr.mul KExpr.one (point generalSelectorIndex)) := by
  rfl

private theorem kMul_c0_lowerAffine_none (left right : KExpr)
    (leftNonconstant : Nonconstant left.c0)
    (rightNonconstant : Nonconstant right.c0) :
    R1CS.lowerAffine (KExpr.mul left right).c0 = none := by
  have first := lowerAffine_mul_eq_none leftNonconstant rightNonconstant
  simp [KExpr.mul, R1CS.lowerAffine, first]

private theorem kMul_c1_lowerAffine_none (left right : KExpr)
    (leftNonconstant : Nonconstant left.c0)
    (rightNonconstant : Nonconstant right.c1) :
    R1CS.lowerAffine (KExpr.mul left right).c1 = none := by
  have first := lowerAffine_mul_eq_none leftNonconstant rightNonconstant
  simp [KExpr.mul, R1CS.lowerAffine, first]

private theorem bitTerm_c0_lowerAffine_none
    (point : Fin productionShape.matrixCount → KExpr) :
    R1CS.lowerAffine
      (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
        bitTerm point).c0 = none := by
  rw [bitTerm_evaluate_eq]
  apply kMul_c0_lowerAffine_none <;> simp [Nonconstant, KExpr.mul]

private theorem bitTerm_c1_lowerAffine_none
    (point : Fin productionShape.matrixCount → KExpr) :
    R1CS.lowerAffine
      (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
        bitTerm point).c1 = none := by
  rw [bitTerm_evaluate_eq]
  apply kMul_c1_lowerAffine_none <;> simp [Nonconstant, KExpr.mul]

private theorem c0_fold_preserves_none
    (point : Fin productionShape.matrixCount → KExpr) :
    ∀ (terms : List (Monomial K productionShape.matrixCount))
      (initial : KExpr),
      R1CS.lowerAffine initial.c0 = none →
      R1CS.lowerAffine
        (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point)) initial).c0 = none
  | [], _, initialNone => initialNone
  | monomial :: terms, initial, initialNone => by
      apply c0_fold_preserves_none point terms
      simp [KExpr.add, R1CS.lowerAffine, initialNone]

private theorem c1_fold_preserves_none
    (point : Fin productionShape.matrixCount → KExpr) :
    ∀ (terms : List (Monomial K productionShape.matrixCount))
      (initial : KExpr),
      R1CS.lowerAffine initial.c1 = none →
      R1CS.lowerAffine
        (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point)) initial).c1 = none
  | [], _, initialNone => initialNone
  | monomial :: terms, initial, initialNone => by
      apply c1_fold_preserves_none point terms
      simp [KExpr.add, R1CS.lowerAffine, initialNone]

private theorem c0_fold_none_of_member
    (point : Fin productionShape.matrixCount → KExpr)
    (target : Monomial K productionShape.matrixCount)
    (targetNone : R1CS.lowerAffine
      (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
        target point).c0 = none) :
    ∀ (terms : List (Monomial K productionShape.matrixCount))
      (initial : KExpr), target ∈ terms →
      R1CS.lowerAffine
        (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point)) initial).c0 = none
  | [], _, member => by simp at member
  | monomial :: terms, initial, member => by
      rcases List.mem_cons.mp member with rfl | member
      · apply c0_fold_preserves_none point terms
        simp [KExpr.add, R1CS.lowerAffine, targetNone]
      · exact c0_fold_none_of_member point target targetNone terms
          (KExpr.add initial
            (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
              monomial point)) member

private theorem c1_fold_none_of_member
    (point : Fin productionShape.matrixCount → KExpr)
    (target : Monomial K productionShape.matrixCount)
    (targetNone : R1CS.lowerAffine
      (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
        target point).c1 = none) :
    ∀ (terms : List (Monomial K productionShape.matrixCount))
      (initial : KExpr), target ∈ terms →
      R1CS.lowerAffine
        (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point)) initial).c1 = none
  | [], _, member => by simp at member
  | monomial :: terms, initial, member => by
      rcases List.mem_cons.mp member with rfl | member
      · apply c1_fold_preserves_none point terms
        simp [KExpr.add, R1CS.lowerAffine, targetNone]
      · exact c1_fold_none_of_member point target targetNone terms
          (KExpr.add initial
            (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
              monomial point)) member

private theorem expression_c0_lowerAffine_none
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.lowerAffine (expression relation interface offset).c0 = none := by
  unfold expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate
  apply c0_fold_none_of_member _ bitTerm (bitTerm_c0_lowerAffine_none _)
  exact bitTerm_mem relation

private theorem expression_c1_lowerAffine_none
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.lowerAffine (expression relation interface offset).c1 = none := by
  unfold expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate
  apply c1_fold_none_of_member _ bitTerm (bitTerm_c1_lowerAffine_none _)
  exact bitTerm_mem relation

private theorem directConstraint_sub_add_eq_none_of_none
    (output : Nat) (left right : Expr)
    (recipeNone : R1CS.lowerAffine (left + right) = none) :
    R1CS.directConstraint (Expr.var output - (left + right)) = none := by
  change R1CS.directConstraint
    (.add (.var output) (.mul (.const (-1)) (.add left right))) = none
  cases leftAffine : R1CS.lowerAffine left <;>
    cases rightAffine : R1CS.lowerAffine right <;>
    simp [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.affineConstraint, R1CS.lowerAffine, leftAffine, rightAffine]
      at recipeNone ⊢

private theorem fold_is_add
    (point : Fin productionShape.matrixCount → KExpr) :
    ∀ (terms : List (Monomial K productionShape.matrixCount))
      (initial : KExpr), terms ≠ [] →
      ∃ left right,
        terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point)) initial = KExpr.add left right
  | [], _, nonempty => (nonempty rfl).elim
  | monomial :: terms, initial, _ => by
      cases terms with
      | nil =>
          exact ⟨initial,
            NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
              monomial point, rfl⟩
      | cons next rest =>
          apply fold_is_add point (next :: rest)
            (KExpr.add initial
              (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                monomial point))
          simp

private theorem expression_is_add
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    ∃ left right, expression relation interface offset = KExpr.add left right := by
  unfold expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.expression
    NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate
  apply fold_is_add
  change (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
    ProductionRelation.polynomial).terms ≠ []
  simp [ConstraintPolynomialLift.liftConstraintPolynomial,
    ProductionRelation.polynomial,
    ProductionRelation.SelectivePolynomial.polynomial,
    ProductionRelation.SelectivePolynomial.terms,
    ProductionRelation.SelectivePolynomial.baseTerms]

private theorem c0_directConstraint_eq_none
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.directConstraint
      (Expr.var offset - (expression relation interface offset).c0) = none := by
  rcases expression_is_add relation interface offset with ⟨left, right, equals⟩
  have recipeNone := expression_c0_lowerAffine_none relation interface offset
  rw [equals] at recipeNone
  rw [equals]
  apply directConstraint_sub_add_eq_none_of_none
  simpa [KExpr.add] using recipeNone

private theorem c1_directConstraint_eq_none
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.directConstraint
      (Expr.var (offset + 1) - (expression relation interface offset).c1) =
        none := by
  rcases expression_is_add relation interface offset with ⟨left, right, equals⟩
  have recipeNone := expression_c1_lowerAffine_none relation interface offset
  rw [equals] at recipeNone
  rw [equals]
  apply directConstraint_sub_add_eq_none_of_none
  simpa [KExpr.add] using recipeNone

@[simp] private theorem mulCount_sub (left right : Expr) :
    R1CS.mulCount (left - right) =
      R1CS.mulCount left + R1CS.mulCount right + 1 := by
  change R1CS.mulCount
    (.add left (.mul (.const (-1)) right)) = _
  simp [R1CS.mulCount]
  omega

private theorem c0_freshCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.constraintFreshCount
      (Expr.var offset - (expression relation interface offset).c0) = 10433 := by
  unfold R1CS.constraintFreshCount
  rw [c0_directConstraint_eq_none relation interface offset, mulCount_sub,
    (expression_mulCounts relation interface offset inputs).1]
  rfl

private theorem c1_freshCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.constraintFreshCount
      (Expr.var (offset + 1) - (expression relation interface offset).c1) =
        10359 := by
  unfold R1CS.constraintFreshCount
  rw [c1_directConstraint_eq_none relation interface offset, mulCount_sub,
    (expression_mulCounts relation interface offset inputs).2]
  rfl

private theorem c0_rowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.constraintRowCount
      (Expr.var offset - (expression relation interface offset).c0) = 10434 := by
  unfold R1CS.constraintRowCount
  rw [c0_directConstraint_eq_none relation interface offset, mulCount_sub,
    (expression_mulCounts relation interface offset inputs).1]
  rfl

private theorem c1_rowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.constraintRowCount
      (Expr.var (offset + 1) - (expression relation interface offset).c1) =
        10360 := by
  unfold R1CS.constraintRowCount
  rw [c1_directConstraint_eq_none relation interface offset, mulCount_sub,
    (expression_mulCounts relation interface offset inputs).2]
  rfl

private theorem core_totalFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 20792 := by
  unfold Formal.ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  change R1CS.totalFreshCount
    (recipeConstraints offset
      [(expression relation interface offset).c0,
        (expression relation interface offset).c1]) = 20792
  simp only [recipeConstraints, R1CS.totalFreshCount, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_freshCount_eq relation interface offset inputs,
    c1_freshCount_eq relation interface offset inputs]

private theorem core_totalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 20794 := by
  unfold Formal.ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  change R1CS.totalRowCount
    (recipeConstraints offset
      [(expression relation interface offset).c0,
        (expression relation interface offset).c1]) = 20794
  simp only [recipeConstraints, R1CS.totalRowCount, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  rw [c0_rowCount_eq relation interface offset inputs,
    c1_rowCount_eq relation interface offset inputs]

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.CircuitFootprint (Formal.ccsCircuit relation interface) where
  freshColumnCount := fun _ => 20792
  physicalRowCount := fun _ => 20794
  freshColumnCount_eq := by
    intro offset
    exact core_totalFreshCount relation interface offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    exact core_totalRowCount relation interface offset (inputs offset)

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 20792 :=
  (footprint relation interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 20794 :=
  (footprint relation interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.ccsCircuit relation interface).main offset)) = 20794 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) = 2 := by
    unfold Formal.ccsCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.localLength_eq
        relation (Formal.ccsInterface relation interface) offset
  rw [logicalColumns, freshColumnCount_eq relation interface inputs offset]

theorem output_linear
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.output relation
        (Formal.ccsInterface relation interface) offset) := by
  refine ⟨rfl, rfl, ?_, ?_⟩
  · intro value equality
    change Expr.var offset = Expr.const value at equality
    cases equality
  · intro value equality
    change Expr.var (offset + 1) = Expr.const value at equality
    cases equality

def logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (Formal.ccsCircuit relation interface).main offset)

def plan
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints relation interface offset
  firstFresh := offset +
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List R1CS.Row :=
  (plan relation interface offset).rows

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation interface offset)

private theorem logicalConstraints_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env) :
    ∀ expression ∈ logicalConstraints relation interface offset,
      expression.VarsBelow (offset +
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount) := by
  have scope :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.flatConstraints_varsBelow
      relation (Formal.ccsInterface relation interface) offset assumptions
  simpa [logicalConstraints,
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount] using scope

theorem physical_implies_logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (physical : PhysicalHolds relation interface offset env) :
    ConstraintsHold env (logicalConstraints relation interface offset) := by
  unfold PhysicalHolds physicalRows at physical
  exact R1CS.lowerConstraints_sound env
    (logicalConstraints relation interface offset)
    (offset +
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount)
    physical

theorem physical_implies_spec
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env)
    (physical : PhysicalHolds relation interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.SpecHolds relation
      (Formal.ccsInterface relation interface) offset env := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.soundness relation
    (Formal.ccsInterface relation interface) env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints relation interface offset)
  exact physical_implies_logicalConstraints relation interface offset env
    physical

theorem physical_complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env)
    (specification :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.SpecHolds relation
        (Formal.ccsInterface relation interface) offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount +
            R1CS.totalFreshCount
              (logicalConstraints relation interface offset)) ∧
        PhysicalHolds relation interface offset completed := by
  rcases NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.completeness
      relation (Formal.ccsInterface relation interface) env offset assumptions
      specification with ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount := by
    simpa [Formal.ccsCircuit] using logicalAgrees
  have scope := logicalConstraints_varsBelow relation interface offset
    logicalEnv assumptions
  have logicalHolds : ConstraintsHold logicalEnv
      (logicalConstraints relation interface offset) := by
    exact logicalRows
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints relation interface offset)
      (offset +
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount)
      scope logicalHolds with
    ⟨completed, physicalAgrees, physicalRowsHold⟩
  refine ⟨completed, logicalAgreesFixed.append physicalAgrees, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal
