import NightstreamFPrime.Layout.Multilinear.PointWeightedHorner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, separate `E_A`.
Obligation: Lower
`E_A = eq(r', r) * sum_(i,j,l) gamma^I_A(i,j,l) cf(y'_(i,j))_l`.

Inputs:
- the 26-coordinate verifier-derived point `r'` and prior point `r`;
- verifier-derived `gamma`;
- 12,096 CCS-matrix-family coefficients, with no Pad coefficient.

Outputs:
- the child-owned exact unshifted `Eval_A` terminal term.

Constraint groups:
- point equality: 102 logical columns, 617 fresh columns, 719 rows;
- 12,096-term Horner: 24,190 logical columns, 84,665 fresh columns,
  108,855 rows;
- parent wiring: zero columns and zero rows.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.eval_A_terminal`.

The global `gamma^(k*d)` shift is not part of this leaf. The final-identity
leaf owns that shift.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal

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

/-- Stable physical wire shape for the separate matrix-family terminal. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.Interface)
    (offset : Nat) : Prop where
  roundPoint : ∀ coordinate,
    KExprLinear (interface.roundPoint offset coordinate)
  priorPoint : ∀ coordinate,
    KExprLinear (interface.priorPoint offset coordinate)
  gamma : KExprLinear (interface.gamma offset)
  outputEval_A : ∀ coordinate,
    KExprLinear (interface.outputEval_A offset coordinate)

/-- The owned `Eval_A` result lies below the canonical CCS child start. -/
theorem output_varsBelow_ccs
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) (env : Env)
    (assumptions :
      (Formal.evalACircuit (Formal.atOffset interface parentOffset)).assumptions
        (Formal.evalAOffset interface parentOffset) env) :
    (Formal.evalAOutput (Formal.atOffset interface parentOffset)
      (Formal.ccsOffset interface parentOffset)).VarsBelow
        (Formal.ccsOffset interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  have childAssumptions : EvalATerminal.Assumptions
      (Formal.evalAInterface frozen) (Formal.evalAStart frozen) env := by
    rw [Formal.evalAStart_atOffset interface parentOffset]
    exact assumptions
  have below := EvalATerminal.output_varsBelow
    (Formal.evalAInterface frozen) (Formal.evalAStart frozen) env
    childAssumptions
  have outputEq : Formal.evalAOutput frozen
      (Formal.ccsOffset interface parentOffset) =
      EvalATerminal.output (Formal.evalAInterface frozen)
        (Formal.evalAStart frozen) := by
    rfl
  have boundEq : Formal.ccsOffset interface parentOffset =
      Formal.evalAStart frozen + localLength (Circuit.ops
        (EvalATerminal.circuit (Formal.evalAInterface frozen)).main
        (Formal.evalAStart frozen)) := by
    calc
      Formal.ccsOffset interface parentOffset = Formal.ccsStart frozen :=
        (Formal.ccsStart_atOffset interface parentOffset).symm
      _ = _ := by
        unfold Formal.ccsStart
        rw [EvalATerminal.localLength_eq]
        rfl
  rw [outputEq, boundEq]
  exact below

private theorem cubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem coefficientExprs_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coefficient ∈
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs
        interface offset,
      KExprLinear coefficient := by
  intro coefficient member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs,
    List.mem_map] at member
  rcases member with ⟨coordinate, _, rfl⟩
  exact inputs.outputEval_A coordinate

private theorem coreInputs
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.InputsLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coreInterface
        interface) offset where
  left := inputs.roundPoint
  right := inputs.priorPoint
  hornerPoint := inputs.gamma
  coefficient := coefficientExprs_linear interface offset inputs

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.circuit interface
        ).main offset)) = 85330 := by
  calc
    _ = (24 * productionShape.cubeVariables - 7) +
        7 * ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs
          interface offset).length - 1) := by
      simpa only [
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.circuit] using
        NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.totalFreshCount
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coreInterface
            interface) cubeVariables_positive offset
          (coreInputs interface offset inputs)
    _ = 85330 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs_length]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.circuit interface
        ).main offset)) = 109630 := by
  calc
    _ = (28 * productionShape.cubeVariables - 9) +
        9 * ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs
          interface offset).length - 1) := by
      simpa only [
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.circuit] using
        NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.totalRowCount
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coreInterface
            interface) cubeVariables_positive offset
          (coreInputs interface offset inputs)
    _ = 109630 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs_length]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

/-- Exact parent-facing physical footprint for separate `Eval_A`. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalAInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.evalACircuit interface) where
  freshColumnCount := fun _ => 85330
  physicalRowCount := fun _ => 109630
  freshColumnCount_eq := by
    intro offset
    unfold Formal.evalACircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.evalACircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalAInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.evalACircuit interface).main offset)) = 85330 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalAInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.evalACircuit interface).main offset)) = 109630 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalAInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.evalACircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.evalACircuit interface).main offset)) = 109630 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.evalACircuit interface).main offset) =
        24300 := by
    unfold Formal.evalACircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.localLength_eq
        (Formal.evalAInterface interface) offset
  rw [logicalColumns, freshColumnCount_eq interface inputs offset]

/-- Exact unmaterialized product shape exported to the final identity. -/
theorem output_shape
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.evalAInterface interface)
      (Formal.evalAStart interface)) :
    NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.ProductOutputShape
      (Formal.evalAOutput interface offset) := by
  unfold Formal.evalAOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.output
  apply NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.output_shape
  · exact cubeVariables_positive
  · exact coreInputs (Formal.evalAInterface interface)
      (Formal.evalAStart interface) inputs
  · intro empty
    have coefficientExprsEmpty :
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs
          (Formal.evalAInterface interface) (Formal.evalAStart interface) =
          [] := by
      simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coreInterface]
        using empty
    have length :=
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.coefficientExprs_length
        (Formal.evalAInterface interface) (Formal.evalAStart interface)
    rw [coefficientExprsEmpty] at length
    simp at length

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal
