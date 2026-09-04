import NightstreamFPrime.Layout.Multilinear.PointEquality
import NightstreamFPrime.Layout.Polynomial.Power
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, `v = Q(r')`.
Obligation: Lower
`v = E_K + gamma^864 E_A + gamma^12960 eq(r', alpha) (F + gamma N)`.

Inputs:
- the 28-coordinate verifier-derived points `r'` and `alpha`;
- verifier-derived `gamma`;
- separate `E_K` and `E_A`, `F`, `N`, and terminal `v` values.

Outputs:
- one exact physical packet for the complete final-identity leaf.

Constraint groups:
- one opaque PointEquality child;
- two opaque fixed-exponent Power children;
- two exact terminal equality assertions.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.final_identity`.

The terminal assertion costs are exact because every preceding child exports
its proved syntactic output shape. The parent does not unfold those children.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stable physical wire shape for the fixed point and Power children. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : Prop where
  roundPoint : ∀ coordinate,
    KExprLinear (interface.roundPoint offset coordinate)
  alpha : ∀ coordinate,
    KExprLinear (interface.alpha offset coordinate)
  gamma : KExprLinear (interface.gamma offset)

/-- Opaque output-shape contracts supplied by the preceding PiCCS children. -/
structure TerminalInputShapes
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : Prop where
  terminal_c0_mulCount : R1CS.mulCount (interface.terminal offset).c0 = 2558
  terminal_c1_mulCount : R1CS.mulCount (interface.terminal offset).c1 = 2557
  eval_K_c0_mulCount : R1CS.mulCount (interface.eval_K offset).c0 = 3
  eval_K_c1_mulCount : R1CS.mulCount (interface.eval_K offset).c1 = 2
  eval_K_c0_nonAffine : R1CS.lowerAffine (interface.eval_K offset).c0 = none
  eval_K_c1_nonAffine : R1CS.lowerAffine (interface.eval_K offset).c1 = none
  eval_A_c0_mulCount : R1CS.mulCount (interface.eval_A offset).c0 = 3
  eval_A_c1_mulCount : R1CS.mulCount (interface.eval_A offset).c1 = 2
  ccs : KExprLinear (interface.ccs offset)
  norm_c0_mulCount : R1CS.mulCount (interface.norm offset).c0 = 10
  norm_c1_mulCount : R1CS.mulCount (interface.norm offset).c1 = 9

private theorem cubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem pointEqualityOutput_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointEqualityOutput
        interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointEqualityOutput
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.output
    NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.program
  apply NightstreamFPrime.Layout.Multilinear.PointEquality.compile_output_linear_of_nonempty
  intro empty
  have lengthZero := congrArg List.length empty
  simp [NightstreamFPrime.Gadgets.Multilinear.PointEquality.Owned.coordinateExprs,
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.canonicalFinIndices_length,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
    at lengthZero

private theorem matrixPowerOutput_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.gammaMatrixOutput
        interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.gammaMatrixOutput
    NightstreamFPrime.Gadgets.Polynomial.Power.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
    NightstreamFPrime.Gadgets.Polynomial.Power.hornerInterface
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent_eq]
  exact NightstreamFPrime.Layout.Polynomial.Power.compile_output_linear_succ
    _ _ 863

private theorem constraintPowerOutput_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.gammaConstraintOutput
        interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.gammaConstraintOutput
    NightstreamFPrime.Gadgets.Polynomial.Power.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
    NightstreamFPrime.Gadgets.Polynomial.Power.hornerInterface
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent_eq]
  exact NightstreamFPrime.Layout.Polynomial.Power.compile_output_linear_succ
    _ _ 12959

private theorem terminalExpr_mulCounts
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    R1CS.mulCount
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0 = 105 ∧
      R1CS.mulCount
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1 = 102 := by
  have pointLinear := pointEqualityOutput_linear interface offset
  have matrixLinear := matrixPowerOutput_linear interface offset
  have constraintLinear := constraintPowerOutput_linear interface offset
  constructor <;>
    simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr,
      KExpr.add, KExpr.mul, R1CS.mulCount,
      inputs.gamma.c0_mulCount, inputs.gamma.c1_mulCount,
      pointLinear.c0_mulCount, pointLinear.c1_mulCount,
      matrixLinear.c0_mulCount, matrixLinear.c1_mulCount,
      constraintLinear.c0_mulCount, constraintLinear.c1_mulCount,
      shapes.eval_K_c0_mulCount, shapes.eval_K_c1_mulCount,
      shapes.eval_A_c0_mulCount, shapes.eval_A_c1_mulCount,
      shapes.ccs.c0_mulCount, shapes.ccs.c1_mulCount,
      shapes.norm_c0_mulCount, shapes.norm_c1_mulCount]

private theorem terminalExpr_nonAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (shapes : TerminalInputShapes interface offset) :
    R1CS.lowerAffine
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0 = none ∧
      R1CS.lowerAffine
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1 = none := by
  constructor <;>
    simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr,
      KExpr.add, R1CS.lowerAffine, shapes.eval_K_c0_nonAffine,
      shapes.eval_K_c1_nonAffine]

private theorem directConstraint_sub_add_eq_none
    (left first second : Expr)
    (firstNone : R1CS.lowerAffine first = none) :
    R1CS.directConstraint (left - (first + second)) = none := by
  change R1CS.directConstraint
    (.add left (.mul (.const (-1)) (.add first second))) = none
  cases left <;>
    simp [R1CS.directConstraint, R1CS.directRecipeRow,
      R1CS.affineConstraint, R1CS.lowerAffine, firstNone]

private theorem terminal_c0_directConstraint_eq_none
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (shapes : TerminalInputShapes interface offset) :
    R1CS.directConstraint
      ((interface.terminal offset).c0 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0) = none := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
  apply directConstraint_sub_add_eq_none
  exact shapes.eval_K_c0_nonAffine

private theorem terminal_c1_directConstraint_eq_none
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (shapes : TerminalInputShapes interface offset) :
    R1CS.directConstraint
      ((interface.terminal offset).c1 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1) = none := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
  apply directConstraint_sub_add_eq_none
  exact shapes.eval_K_c1_nonAffine

private theorem terminal_c0_freshCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    R1CS.constraintFreshCount
      ((interface.terminal offset).c0 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0) = 2664 := by
  unfold R1CS.constraintFreshCount
  rw [terminal_c0_directConstraint_eq_none interface offset shapes]
  have counts := terminalExpr_mulCounts interface offset inputs shapes
  change R1CS.mulCount
    (.add (interface.terminal offset).c0
      (.mul (.const (-1))
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0)) = 2664
  simp only [R1CS.mulCount, shapes.terminal_c0_mulCount, counts.1]

private theorem terminal_c1_freshCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    R1CS.constraintFreshCount
      ((interface.terminal offset).c1 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1) = 2660 := by
  unfold R1CS.constraintFreshCount
  rw [terminal_c1_directConstraint_eq_none interface offset shapes]
  have counts := terminalExpr_mulCounts interface offset inputs shapes
  change R1CS.mulCount
    (.add (interface.terminal offset).c1
      (.mul (.const (-1))
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1)) = 2660
  simp only [R1CS.mulCount, shapes.terminal_c1_mulCount, counts.2]

private theorem terminal_c0_rowCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    R1CS.constraintRowCount
      ((interface.terminal offset).c0 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c0) = 2665 := by
  unfold R1CS.constraintRowCount
  rw [terminal_c0_directConstraint_eq_none interface offset shapes]
  have counts := terminalExpr_mulCounts interface offset inputs shapes
  change R1CS.mulCount
      (.add (interface.terminal offset).c0
        (.mul (.const (-1))
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
            interface offset).c0)) + 1 = 2665
  simp only [R1CS.mulCount, shapes.terminal_c0_mulCount, counts.1]

private theorem terminal_c1_rowCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    R1CS.constraintRowCount
      ((interface.terminal offset).c1 -
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
          interface offset).c1) = 2661 := by
  unfold R1CS.constraintRowCount
  rw [terminal_c1_directConstraint_eq_none interface offset shapes]
  have counts := terminalExpr_mulCounts interface offset inputs shapes
  change R1CS.mulCount
      (.add (interface.terminal offset).c1
        (.mul (.const (-1))
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalExpr
            interface offset).c1)) + 1 = 2661
  simp only [R1CS.mulCount, shapes.terminal_c1_mulCount, counts.2]

private theorem pointInputs
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coordinate,
      KExprLinear
          ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointInterfaceAt
            interface offset).left offset coordinate) ∧
        KExprLinear
          ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointInterfaceAt
            interface offset).right offset coordinate) := by
  intro coordinate
  exact ⟨inputs.roundPoint coordinate, inputs.alpha coordinate⟩

private theorem point_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
        interface offset).main offset)) = 665 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
  calc
    _ = 24 * productionShape.cubeVariables - 7 :=
      NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalFreshCount_of_positive
        _ offset cubeVariables_positive (pointInputs interface offset inputs)
    _ = 665 := by
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem point_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
        interface offset).main offset)) = 775 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
  calc
    _ = 28 * productionShape.cubeVariables - 9 :=
      NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalRowCount_of_positive
        _ offset cubeVariables_positive (pointInputs interface offset inputs)
    _ = 775 := by
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem matrixPower_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixPowerCircuitAt
        interface offset).main
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixOffset
          interface offset))) = 6041 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixPowerCircuitAt
  calc
    _ = 7 *
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent -
          1) :=
      NightstreamFPrime.Layout.Polynomial.Power.ownedCircuit_totalFreshCount
        _ _ _ inputs.gamma
    _ = 6041 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent_eq]

private theorem matrixPower_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixPowerCircuitAt
        interface offset).main
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixOffset
          interface offset))) = 7769 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixPowerCircuitAt
  calc
    _ = if NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent =
          0 then 0 else
        9 * NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent -
          7 :=
      NightstreamFPrime.Layout.Polynomial.Power.ownedCircuit_totalRowCount
        _ _ _ inputs.gamma
    _ = 7769 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.matrixExponent_eq]
      norm_num

private theorem constraintPower_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintPowerCircuitAt
        interface offset).main
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintOffset
          interface offset))) = 90713 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintPowerCircuitAt
  calc
    _ = 7 *
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent -
          1) :=
      NightstreamFPrime.Layout.Polynomial.Power.ownedCircuit_totalFreshCount
        _ _ _ inputs.gamma
    _ = 90713 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent_eq]

private theorem constraintPower_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintPowerCircuitAt
        interface offset).main
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintOffset
          interface offset))) = 116633 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintPowerCircuitAt
  calc
    _ = if
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent =
          0 then 0 else
        9 *
            NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent -
          7 :=
      NightstreamFPrime.Layout.Polynomial.Power.ownedCircuit_totalRowCount
        _ _ _ inputs.gamma
    _ = 116633 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.constraintExponent_eq]
      norm_num

def terminalFreshColumnCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : Nat :=
  R1CS.totalFreshCount
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalAssertions
      interface offset)

def terminalPhysicalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : Nat :=
  R1CS.totalRowCount
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalAssertions
      interface offset)

/-- Exact fresh-column cost of the two v1_1 terminal equality components. -/
theorem terminalFreshColumnCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    terminalFreshColumnCount interface offset = 5324 := by
  unfold terminalFreshColumnCount
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalAssertions
    KExpr.equalities R1CS.totalFreshCount
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Nat.add_zero]
  rw [terminal_c0_freshCount_eq interface offset inputs shapes,
    terminal_c1_freshCount_eq interface offset inputs shapes]

/-- Exact physical-row cost of the two v1_1 terminal equality components. -/
theorem terminalPhysicalRowCount_eq
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (shapes : TerminalInputShapes interface offset) :
    terminalPhysicalRowCount interface offset = 5326 := by
  unfold terminalPhysicalRowCount
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.terminalAssertions
    KExpr.equalities R1CS.totalRowCount
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Nat.add_zero]
  rw [terminal_c0_rowCount_eq interface offset inputs shapes,
    terminal_c1_rowCount_eq interface offset inputs shapes]

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit interface
        ).main offset)) =
      97419 + terminalFreshColumnCount interface offset := by
  change R1CS.totalFreshCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.opsAt
      interface offset)) = _
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.flatConstraints_opsAt]
  simp only [R1CS.totalFreshCount_append]
  rw [point_totalFreshCount interface offset inputs,
    matrixPower_totalFreshCount interface offset inputs,
    constraintPower_totalFreshCount interface offset inputs]
  rfl

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit interface
        ).main offset)) =
      125177 + terminalPhysicalRowCount interface offset := by
  change R1CS.totalRowCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.opsAt
      interface offset)) = _
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.flatConstraints_opsAt]
  simp only [R1CS.totalRowCount_append]
  rw [point_totalRowCount interface offset inputs,
    matrixPower_totalRowCount interface offset inputs,
    constraintPower_totalRowCount interface offset inputs]
  rfl

/-- Exact parent-facing footprint for the complete final-identity leaf. -/
def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.finalIdentityInterface relation interface) offset) :
    R1CS.CircuitFootprint (Formal.finalIdentityCircuit relation interface) where
  freshColumnCount := fun offset =>
    97419 + terminalFreshColumnCount
      (Formal.finalIdentityInterface relation interface) offset
  physicalRowCount := fun offset =>
    125177 + terminalPhysicalRowCount
      (Formal.finalIdentityInterface relation interface) offset
  freshColumnCount_eq := by
    intro offset
    unfold Formal.finalIdentityCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.finalIdentityCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.finalIdentityInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.finalIdentityCircuit relation interface).main offset)) =
      97419 + terminalFreshColumnCount
        (Formal.finalIdentityInterface relation interface) offset :=
  (footprint relation interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.finalIdentityInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.finalIdentityCircuit relation interface).main offset)) =
      125177 + terminalPhysicalRowCount
        (Formal.finalIdentityInterface relation interface) offset :=
  (footprint relation interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.finalIdentityInterface relation interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops
        (Formal.finalIdentityCircuit relation interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.finalIdentityCircuit relation interface).main offset)) =
      125177 + terminalFreshColumnCount
        (Formal.finalIdentityInterface relation interface) offset := by
  have logicalColumns : localLength (Circuit.ops
      (Formal.finalIdentityCircuit relation interface).main offset) = 27758 := by
    exact (Formal.finalIdentityCircuit relation interface).privateCount_eq offset
  rw [logicalColumns, freshColumnCount_eq relation interface inputs offset]
  omega

/-- Exact logical rows of the final-identity leaf, before generic R1CS
lowering. -/
def logicalConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.main interface)
      offset)

/-- Multiplication intermediates start after the exact logical leaf interval. -/
def plan
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + 27758

def physicalRows
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : List R1CS.Row :=
  (R1CS.lowerConstraints (logicalConstraints interface offset)
    (offset + 27758)).rows

def PhysicalHolds
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows interface offset)

private theorem logicalConstraints_varsBelow
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Assumptions
        interface offset env) :
    ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + 27758) := by
  have scope :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.flatConstraints_varsBelow
      interface offset env assumptions
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.localLength_eq]
    at scope
  simpa [logicalConstraints,
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit] using scope

theorem physical_implies_logicalConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (env : Env)
    (physical : PhysicalHolds interface offset env) :
    ConstraintsHold env (logicalConstraints interface offset) := by
  unfold PhysicalHolds physicalRows at physical
  exact R1CS.lowerConstraints_sound env
    (logicalConstraints interface offset) (offset + 27758) physical

theorem physical_implies_spec
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Assumptions
        interface offset env)
    (physical : PhysicalHolds interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.SpecHolds
      interface offset env := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.soundness
    interface env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact physical_implies_logicalConstraints interface offset env physical

/-- Honest logical execution followed by the generic R1CS executor constructs
all logical and multiplication-witness columns. -/
theorem physical_complete
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Assumptions
        interface offset env)
    (specification :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.SpecHolds
        interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (27758 + R1CS.totalFreshCount
            (logicalConstraints interface offset)) ∧
        PhysicalHolds interface offset completed := by
  rcases NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.completeness
      interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have lengthEq : localLength (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.main interface)
        offset) = 27758 := by
    change localLength (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit interface
        ).main offset) = 27758
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.localLength_eq
      interface offset
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset 27758 := by
    rw [lengthEq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Assumptions
        interface offset logicalEnv := {
    point := assumptions.point
    gammaBelow := assumptions.gammaBelow
    eval_KBelow := assumptions.eval_KBelow
    eval_ABelow := assumptions.eval_ABelow
    ccsBelow := assumptions.ccsBelow
    normBelow := assumptions.normBelow
    terminalBelow := assumptions.terminalBelow
  }
  have scope := logicalConstraints_varsBelow interface offset logicalEnv
    logicalAssumptions
  have logicalHolds :
      ConstraintsHold logicalEnv (logicalConstraints interface offset) := by
    exact logicalRows
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset) (offset + 27758) scope logicalHolds
      with ⟨completed, physicalAgrees, physicalRowsHold⟩
  refine ⟨completed, logicalAgreesFixed.append physicalAgrees, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity
