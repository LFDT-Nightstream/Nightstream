import NightstreamFPrime.Layout.Multilinear.PointEquality
import NightstreamFPrime.Layout.Polynomial.Power
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, `v = Q(r')`.
Obligation: Lower
`v = E_K + gamma^864 E_A + gamma^12960 eq(r', alpha) (F + gamma N)`.

Inputs:
- the 24-coordinate verifier-derived points `r'` and `alpha`;
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

The two terminal assertion costs remain symbolic because `F` contains the
relation-owned sparse CCS expression. The fixed production relation supplies
their numeric corollary when its sparse polynomial is fixed.
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

private theorem cubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

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
        interface offset).main offset)) = 569 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
  calc
    _ = 24 * productionShape.cubeVariables - 7 :=
      NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalFreshCount_of_positive
        _ offset cubeVariables_positive (pointInputs interface offset inputs)
    _ = 569 := by
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem point_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
        interface offset).main offset)) = 663 := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointCircuitAt
  calc
    _ = 28 * productionShape.cubeVariables - 9 :=
      NightstreamFPrime.Layout.Multilinear.PointEquality.ownedCircuit_totalRowCount_of_positive
        _ offset cubeVariables_positive (pointInputs interface offset inputs)
    _ = 663 := by
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

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit interface
        ).main offset)) =
      97323 + terminalFreshColumnCount interface offset := by
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
      125065 + terminalPhysicalRowCount interface offset := by
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
    97323 + terminalFreshColumnCount
      (Formal.finalIdentityInterface relation interface) offset
  physicalRowCount := fun offset =>
    125065 + terminalPhysicalRowCount
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
      97323 + terminalFreshColumnCount
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
      125065 + terminalPhysicalRowCount
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
      125065 + terminalFreshColumnCount
        (Formal.finalIdentityInterface relation interface) offset := by
  have logicalColumns : localLength (Circuit.ops
      (Formal.finalIdentityCircuit relation interface).main offset) = 27742 := by
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
  firstFresh := offset + 27742

def physicalRows
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.Interface)
    (offset : Nat) : List R1CS.Row :=
  (R1CS.lowerConstraints (logicalConstraints interface offset)
    (offset + 27742)).rows

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
      expression.VarsBelow (offset + 27742) := by
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
    (logicalConstraints interface offset) (offset + 27742) physical

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
          (27742 + R1CS.totalFreshCount
            (logicalConstraints interface offset)) ∧
        PhysicalHolds interface offset completed := by
  rcases NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.completeness
      interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have lengthEq : localLength (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.main interface)
        offset) = 27742 := by
    change localLength (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit interface
        ).main offset) = 27742
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.localLength_eq
      interface offset
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset 27742 := by
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
      (logicalConstraints interface offset) (offset + 27742) scope logicalHolds
      with ⟨completed, physicalAgrees, physicalRowsHold⟩
  refine ⟨completed, logicalAgreesFixed.append physicalAgrees, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity
