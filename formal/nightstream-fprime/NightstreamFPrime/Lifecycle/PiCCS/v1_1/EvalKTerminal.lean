import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 4, `E_K`.
Obligation: Enforce
`E_K = eq(r', r) * sum_(i,l) gamma^I_K(i,l) cf(y'_i)_l`.

Inputs:
- the verifier-derived round point `r'` and prior point `r`;
- the verifier-derived challenge `gamma`;
- all output Pad-family coefficients, and no CCS-matrix coefficient;

Outputs:
- the child-owned exact unshifted `Eval_K` terminal term.

Constraint groups:
- C1-C3: the opaque reusable `PointWeightedHorner` circuit.

Parent coverage:
- `ProtocolPolynomial.padAtMessage` in the production PiCCS terminal.

This file owns only the v1.1 Pad coordinate order and production-key wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure Interface where
  roundPoint : Nat → Fin productionShape.cubeVariables → KExpr
  priorPoint : Nat → Fin productionShape.cubeVariables → KExpr
  gamma : Nat → KExpr
  outputEval_K : Nat → PadCoordinate productionShape → KExpr

def coefficientExprs (interface : Interface) (offset : Nat) : List KExpr :=
  (canonicalPadCoordinates productionShape).map (interface.outputEval_K offset)

def coreInterface (interface : Interface) :
    PointWeightedHorner.Owned.Interface productionShape.cubeVariables where
  left := interface.roundPoint
  right := interface.priorPoint
  hornerPoint := interface.gamma
  coefficients := coefficientExprs interface

/-- Child-owned `E_K`. -/
def output (interface : Interface) (offset : Nat) : KExpr :=
  PointWeightedHorner.Owned.output (coreInterface interface) offset

def pointInterfaceAt (interface : Interface) (parentOffset : Nat) :=
  PointWeightedHorner.Owned.pointInterfaceAt
    (coreInterface interface) parentOffset

def hornerInterfaceAt (interface : Interface) (parentOffset : Nat) :=
  PointWeightedHorner.Owned.hornerInterfaceAt
    (coreInterface interface) parentOffset

def pointLength (interface : Interface) (offset : Nat) : Nat :=
  PointWeightedHorner.Owned.pointLength (coreInterface interface) offset

def hornerLength (interface : Interface) (offset : Nat) : Nat :=
  PointWeightedHorner.Owned.hornerLength (coreInterface interface) offset

private theorem cubeVariables_positive : 0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  PointWeightedHorner.Owned.Assumptions (coreInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  PointWeightedHorner.Owned.SpecHolds (coreInterface interface) offset env

def circuit (interface : Interface) : FormalCircuit :=
  PointWeightedHorner.Owned.circuit
    (coreInterface interface) cubeVariables_positive

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  PointWeightedHorner.Owned.soundness (coreInterface interface) env offset
    assumptions rows

/-- Honest execution constructs all three derived values. -/
theorem build (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  PointWeightedHorner.Owned.build (coreInterface interface) env offset
    cubeVariables_positive assumptions

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  build interface env offset assumptions

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  PointWeightedHorner.Owned.flatConstraints_varsBelow
    (coreInterface interface)
    cubeVariables_positive offset env assumptions

/-- The owned `Eval_K` result lies inside this leaf's declared interval. -/
theorem output_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (output interface offset).VarsBelow
      (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  PointWeightedHorner.Owned.output_varsBelow (coreInterface interface)
    cubeVariables_positive offset env assumptions

theorem coefficientExprs_length (interface : Interface) (offset : Nat) :
    (coefficientExprs interface offset).length = 864 := by
  simp [coefficientExprs, canonicalPadCoordinates_length, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape,
    Shape.padEvaluationCount, ringDegree]

theorem pointLength_eq (interface : Interface) (offset : Nat) :
    pointLength interface offset = 110 := by
  unfold pointLength
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
    PointWeightedHorner.Owned.pointLength_eq_of_positive
      (coreInterface interface) offset cubeVariables_positive

theorem hornerLength_eq (interface : Interface) (offset : Nat) :
    hornerLength interface offset = 1726 := by
  unfold hornerLength
  rw [PointWeightedHorner.Owned.hornerLength_eq]
  change 2 * ((coefficientExprs interface offset).length - 1) = 1726
  rw [coefficientExprs_length]

def privateCount : Nat := 1836

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 1836 := by
  unfold circuit
  rw [PointWeightedHorner.Owned.localLength_eq]
  change (4 * productionShape.cubeVariables - 2) +
    2 * ((coefficientExprs interface offset).length - 1) = 1836
  rw [coefficientExprs_length]
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 2 :=
  PointWeightedHorner.Owned.operations_length (coreInterface interface)
    cubeVariables_positive offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      1836 := by
  unfold circuit
  rw [PointWeightedHorner.Owned.flatConstraints_length]
  change (4 * productionShape.cubeVariables - 2) +
    2 * ((coefficientExprs interface offset).length - 1) = 1836
  rw [coefficientExprs_length]
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

/-- Concrete parent coverage: the generic point-weighted Horner value is
exactly production `ProtocolPolynomial.padAtMessage`. -/
theorem spec_implies_keyPadAtMessage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface) (offset : Nat) (env : Env)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (pointInterfaceAt interface offset) offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint)
    (priorPointEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface offset) offset env =
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input.priorPoint)
    (gammaEq : (interface.gamma offset).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.gamma)
    (outputEvalEq : ∀ coordinate,
      (interface.outputEval_K offset coordinate).eval env =
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output.padImage coordinate)
    (specification : SpecHolds interface offset env) :
    (output interface offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output := by
  let input := (ChallengeDerivation.productionContext
    relation ajtai running fresh).input
  let execution := (ProductionKey.key relation ajtai).piCcsExecution
    running fresh proof
  let message := ((ProductionKey.key relation ajtai).piCcsCertificate
    running fresh proof).output
  have coefficientsEq :
      (coefficientExprs interface offset).map (KExpr.eval env) =
        NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputPadCoefficientList
          message := by
    unfold coefficientExprs
      NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputPadCoefficientList
    rw [List.map_map]
    apply List.map_congr_left
    intro coordinate _
    exact outputEvalEq coordinate
  have pointEq := specification.point
  unfold PointEquality.Owned.SpecHolds at pointEq
  change PointEquality.Owned.evalLeftPoint
      (PointWeightedHorner.Owned.pointInterfaceAt
        (coreInterface interface) offset)
        offset env = execution.coins.roundPoint at roundPointEq
  change PointEquality.Owned.evalRightPoint
      (PointWeightedHorner.Owned.pointInterfaceAt
        (coreInterface interface) offset)
        offset env = input.priorPoint at priorPointEq
  rw [roundPointEq, priorPointEq] at pointEq
  have pointValueEq :
      (PointWeightedHorner.Owned.pointOutput
        (coreInterface interface) offset).eval env =
      SumCheckTruthPath.pointEquality extensionOps
        execution.coins.roundPoint input.priorPoint := by
    simpa [pointInterfaceAt, PointWeightedHorner.Owned.pointOutput,
      PointWeightedHorner.Owned.pointInterfaceAt,
      coreInterface, input, execution] using pointEq
  have hornerEq := specification.horner
  unfold Horner.Owned.SpecHolds at hornerEq
  change (PointWeightedHorner.Owned.weightedSum
    (coreInterface interface) offset).eval env =
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
      ((interface.gamma offset).eval env)
      ((coefficientExprs interface offset).map (KExpr.eval env)) at hornerEq
  rw [gammaEq, coefficientsEq] at hornerEq
  have hornerValueEq :
      (PointWeightedHorner.Owned.weightedSum
        (coreInterface interface) offset).eval env =
      SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
        execution.coins.gamma
        (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputPadCoefficientList
          message) := by
    simpa [execution, message] using hornerEq
  calc
    (output interface offset).eval env =
        K.mul
          ((PointWeightedHorner.Owned.pointOutput
            (coreInterface interface) offset).eval env)
          ((PointWeightedHorner.Owned.weightedSum
            (coreInterface interface) offset).eval env) := by
      rfl
    _ = extensionOps.mul
        (SumCheckTruthPath.pointEquality extensionOps
          execution.coins.roundPoint input.priorPoint)
        (SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
          execution.coins.gamma
          (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.outputPadCoefficientList
            message)) := by
      rw [pointValueEq, hornerValueEq]
      rfl
    _ = ProtocolPolynomial.padAtMessage extensionOps input
        execution.coins.gamma execution.coins.roundPoint message :=
      (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.padAtMessage_eq_pointEquality_mul_horner
        extensionOps extensionLaws input execution.coins.gamma
        execution.coins.roundPoint message).symm

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal
