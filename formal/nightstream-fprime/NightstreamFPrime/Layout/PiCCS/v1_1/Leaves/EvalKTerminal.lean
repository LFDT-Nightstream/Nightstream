import NightstreamFPrime.Layout.Multilinear.PointWeightedHorner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Step 4, separate `E_K`.
Obligation: Lower
`E_K = eq(r', r) * sum_(i,l) gamma^I_K(i,l) cf(y'_i)_l`.

Inputs:
- the 26-coordinate verifier-derived point `r'` and prior point `r`;
- verifier-derived `gamma`;
- 864 Pad-family coefficients, with no CCS-matrix coefficient.

Outputs:
- the child-owned exact unshifted `Eval_K` terminal term.

Constraint groups:
- point equality: 102 logical columns, 617 fresh columns, 719 rows;
- 864-term Horner: 1,726 logical columns, 6,041 fresh columns,
  7,767 rows;
- parent wiring: zero columns and zero rows.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.eval_K_terminal`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal

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

/-- Stable physical wire shape for the separate Pad-family terminal. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.Interface)
    (offset : Nat) : Prop where
  roundPoint : ∀ coordinate,
    KExprLinear (interface.roundPoint offset coordinate)
  priorPoint : ∀ coordinate,
    KExprLinear (interface.priorPoint offset coordinate)
  gamma : KExprLinear (interface.gamma offset)
  outputEval_K : ∀ coordinate,
    KExprLinear (interface.outputEval_K offset coordinate)

/-- The owned `Eval_K` result lies below the canonical `Eval_A` child start. -/
theorem output_varsBelow_evalA
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) (env : Env)
    (assumptions :
      (Formal.evalKCircuit (Formal.atOffset interface parentOffset)).assumptions
        (Formal.evalKOffset interface parentOffset) env) :
    (Formal.evalKOutput (Formal.atOffset interface parentOffset)
      (Formal.evalAOffset interface parentOffset)).VarsBelow
        (Formal.evalAOffset interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  have childAssumptions : EvalKTerminal.Assumptions
      (Formal.evalKInterface frozen) (Formal.evalKStart frozen) env := by
    rw [Formal.evalKStart_atOffset interface parentOffset]
    exact assumptions
  have below := EvalKTerminal.output_varsBelow
    (Formal.evalKInterface frozen) (Formal.evalKStart frozen) env
    childAssumptions
  have outputEq : Formal.evalKOutput frozen
      (Formal.evalAOffset interface parentOffset) =
      EvalKTerminal.output (Formal.evalKInterface frozen)
        (Formal.evalKStart frozen) := by
    rfl
  have boundEq : Formal.evalAOffset interface parentOffset =
      Formal.evalKStart frozen + localLength (Circuit.ops
        (EvalKTerminal.circuit (Formal.evalKInterface frozen)).main
        (Formal.evalKStart frozen)) := by
    calc
      Formal.evalAOffset interface parentOffset = Formal.evalAStart frozen :=
        (Formal.evalAStart_atOffset interface parentOffset).symm
      _ = _ := by
        unfold Formal.evalAStart
        rw [EvalKTerminal.localLength_eq]
        rfl
  rw [outputEq, boundEq]
  exact below

private theorem cubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem coefficientExprs_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coefficient ∈
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs
        interface offset,
      KExprLinear coefficient := by
  intro coefficient member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs,
    List.mem_map] at member
  rcases member with ⟨coordinate, _, rfl⟩
  exact inputs.outputEval_K coordinate

private theorem coreInputs
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.InputsLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coreInterface
        interface) offset where
  left := inputs.roundPoint
  right := inputs.priorPoint
  hornerPoint := inputs.gamma
  coefficient := coefficientExprs_linear interface offset inputs

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.circuit interface
        ).main offset)) = 6706 := by
  calc
    _ = (24 * productionShape.cubeVariables - 7) +
        7 * ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs
          interface offset).length - 1) := by
      simpa only [
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.circuit] using
        NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.totalFreshCount
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coreInterface
            interface) cubeVariables_positive offset
          (coreInputs interface offset inputs)
    _ = 6706 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs_length]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.circuit interface
        ).main offset)) = 8542 := by
  calc
    _ = (28 * productionShape.cubeVariables - 9) +
        9 * ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs
          interface offset).length - 1) := by
      simpa only [
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.circuit] using
        NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.totalRowCount
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coreInterface
            interface) cubeVariables_positive offset
          (coreInputs interface offset inputs)
    _ = 8542 := by
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs_length]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

/-- Exact parent-facing physical footprint for separate `Eval_K`. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalKInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.evalKCircuit interface) where
  freshColumnCount := fun _ => 6706
  physicalRowCount := fun _ => 8542
  freshColumnCount_eq := by
    intro offset
    unfold Formal.evalKCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.evalKCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalKInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.evalKCircuit interface).main offset)) = 6706 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalKInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.evalKCircuit interface).main offset)) = 8542 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.evalKInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.evalKCircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.evalKCircuit interface).main offset)) = 8542 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.evalKCircuit interface).main offset) =
        1836 := by
    unfold Formal.evalKCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.localLength_eq
        (Formal.evalKInterface interface) offset
  rw [logicalColumns, freshColumnCount_eq interface inputs offset]

/-- Exact unmaterialized product shape exported to the final identity. -/
theorem output_shape
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.evalKInterface interface)
      (Formal.evalKStart interface)) :
    NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.ProductOutputShape
      (Formal.evalKOutput interface offset) := by
  unfold Formal.evalKOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.output
  apply NightstreamFPrime.Layout.Multilinear.PointWeightedHorner.output_shape
  · exact cubeVariables_positive
  · exact coreInputs (Formal.evalKInterface interface)
      (Formal.evalKStart interface) inputs
  · intro empty
    have coefficientExprsEmpty :
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs
          (Formal.evalKInterface interface) (Formal.evalKStart interface) =
          [] := by
      simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coreInterface]
        using empty
    have length :=
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.coefficientExprs_length
        (Formal.evalKInterface interface) (Formal.evalKStart interface)
    rw [coefficientExprsEmpty] at length
    simp at length

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal
