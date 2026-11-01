import NightstreamFPrime.Gadgets.Multilinear.PointEquality
import NightstreamFPrime.Gadgets.Polynomial.Power
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.2, Section 7.3, Step 4, `v = Q(r')`.
Obligation: Enforce
`E_K + gamma^864 E_A + gamma^4320 eq(r',alpha) (F + gamma N)`.

Inputs:
- verifier-derived `r'`, `alpha`, and `gamma`;
- the already-constrained `E_K`, `E_A`, `F`, and `N` leaf outputs;
- the SumCheck terminal claim `v`.

Outputs:
- the exact complete v1.2 PiCCS terminal equality.

Constraint groups:
- C1: one opaque owned `PointEquality` child;
- C2: one opaque owned `Power` child for `gamma^864`;
- C3: one opaque owned `Power` child that raises `gamma^864` to `t + 1 = 5`,
  giving `gamma^4320`;
- C4: two extension-component final-identity assertions.

Parent coverage:
- `ProtocolPolynomial.terminalFromMessage` in `PiCCS.v1_2.Coverage.chain`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The `γ` exponent of `E_A`: the pad-evaluation count of the shape. -/
def matrixExponentOf (shape : Shape) : Nat := shape.matrixEvaluationOffset

/-- The `γ` exponent of the constraint terms: the constraint offset of the shape. -/
def constraintExponentOf (shape : Shape) : Nat := shape.constraintOffset

/-- The circuit computes `γ ^ constraintExponentOf` as `(γ ^ matrixExponentOf) ^ (t + 1)`. -/
def constraintFactorOf (shape : Shape) : Nat := shape.matrixCount + 1

theorem matrixExponent_eq : matrixExponentOf productionShape = 864 := by
  norm_num [matrixExponentOf, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.matrixEvaluationOffset,
    Shape.padEvaluationCount, ringDegree]

theorem constraintExponent_eq : constraintExponentOf productionShape = 4320 := by
  norm_num [constraintExponentOf, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.constraintOffset,
    Shape.padEvaluationCount, Shape.matrixEvaluationCount, ringDegree]

theorem constraintFactor_eq : constraintFactorOf productionShape = 5 := by
  norm_num [constraintFactorOf, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape]

theorem constraintExponent_eq_mul (shape : Shape) :
    constraintExponentOf shape = matrixExponentOf shape * constraintFactorOf shape := by
  simp only [constraintExponentOf, matrixExponentOf, constraintFactorOf,
    Shape.constraintOffset_eq, Shape.matrixEvaluationOffset, Shape.padEvaluationCount]

private theorem power_mul (value : K) (left right : Nat) :
    TargetPolynomial.power extensionOps.toOps value (left * right) =
      TargetPolynomial.power extensionOps.toOps
        (TargetPolynomial.power extensionOps.toOps value left) right := by
  have shiftLaws : TargetPolynomial.ShiftLaws extensionOps.toOps :=
    { one_mul := extensionLaws.one_mul
      mul_assoc := extensionLaws.mul_assoc
      mul_zero := extensionLaws.mul_zero
      mul_add := extensionLaws.left_distrib }
  induction right with
  | zero => rfl
  | succ right inductionHypothesis =>
      rw [show left * (right + 1) = left + left * right by ring,
        TargetPolynomial.power_add extensionOps.toOps shiftLaws,
        inductionHypothesis]
      rfl

theorem gammaFreshPower_of_one {shape : Shape} (fresh : shape.freshCount = 1) (gamma : K) :
    TargetPolynomial.power extensionOps.toOps gamma shape.freshCount = gamma := by
  rw [fresh]
  simpa [TargetPolynomial.power] using extensionLaws.mul_one gamma

theorem gammaFreshPower_eq (gamma : K) :
    TargetPolynomial.power extensionOps.toOps gamma
      productionShape.freshCount = gamma := by
  simpa [productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
    TargetPolynomial.power] using extensionLaws.mul_one gamma

/-- The leaf's inputs for a Π_CCS shape: the production shape, or option B's light shape. -/
structure ShapeInterface (shape : Shape) where
  roundPoint : Nat → Fin shape.cubeVariables → KExpr
  alpha : Nat → Fin shape.cubeVariables → KExpr
  gamma : Nat → KExpr
  eval_K : Nat → KExpr
  eval_A : Nat → KExpr
  ccs : Nat → KExpr
  norm : Nat → KExpr
  terminal : Nat → KExpr

abbrev Interface := ShapeInterface productionShape

section Shape

variable {shape : Shape}

def pointInterfaceAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    PointEquality.Owned.Interface shape.cubeVariables where
  left := fun _ => interface.roundPoint parentOffset
  right := fun _ => interface.alpha parentOffset

def matrixPowerInterfaceAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    Power.Interface where
  point := fun _ => interface.gamma parentOffset

def pointCircuitAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    FormalCircuit :=
  PointEquality.Owned.circuit (pointInterfaceAt interface parentOffset)

def matrixPowerCircuitAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    FormalCircuit :=
  Power.circuit (matrixExponentOf shape) (matrixPowerInterfaceAt interface parentOffset)

def pointLength (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  localLength (Circuit.ops (pointCircuitAt interface offset).main offset)

def matrixOffset (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  offset + pointLength interface offset

def matrixLength (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  localLength (Circuit.ops (matrixPowerCircuitAt interface offset).main
    (matrixOffset interface offset))

def constraintOffset (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  matrixOffset interface offset + matrixLength interface offset

private theorem constraintOffset_eq (interface : ShapeInterface shape) (offset : Nat) :
    constraintOffset interface offset =
      offset + (pointLength interface offset + matrixLength interface offset) := by
  unfold constraintOffset matrixOffset
  exact Nat.add_assoc _ _ _

def gammaMatrixOutput (interface : ShapeInterface shape) (offset : Nat) : KExpr :=
  Power.output (matrixExponentOf shape) (matrixPowerInterfaceAt interface offset)
    (matrixOffset interface offset)

def constraintPowerInterfaceAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    Power.Interface where
  point := fun _ => gammaMatrixOutput interface parentOffset

def constraintPowerCircuitAt (interface : ShapeInterface shape) (parentOffset : Nat) :
    FormalCircuit :=
  Power.circuit (constraintFactorOf shape)
    (constraintPowerInterfaceAt interface parentOffset)

def constraintLength (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  localLength (Circuit.ops (constraintPowerCircuitAt interface offset).main
    (constraintOffset interface offset))

def finalOffset (interface : ShapeInterface shape) (offset : Nat) : Nat :=
  constraintOffset interface offset + constraintLength interface offset

def pointEqualityOutput (interface : ShapeInterface shape) (offset : Nat) : KExpr :=
  PointEquality.Owned.output (pointInterfaceAt interface offset) offset

def gammaConstraintOutput (interface : ShapeInterface shape) (offset : Nat) : KExpr :=
  Power.output (constraintFactorOf shape) (constraintPowerInterfaceAt interface offset)
    (constraintOffset interface offset)

def pointName : String := "piccs.v1_2.final.point_equality"
def matrixPowerName : String := "piccs.v1_2.final.gamma_matrix_offset"
def constraintPowerName : String := "piccs.v1_2.final.gamma_constraint_offset"

def pointSubcircuit (interface : ShapeInterface shape) (offset : Nat) : Subcircuit :=
  (pointCircuitAt interface offset).asSubcircuit pointName offset

def matrixPowerSubcircuit (interface : ShapeInterface shape) (offset : Nat) : Subcircuit :=
  (matrixPowerCircuitAt interface offset).asSubcircuit matrixPowerName
    (matrixOffset interface offset)

def constraintPowerSubcircuit (interface : ShapeInterface shape)
    (offset : Nat) : Subcircuit :=
  (constraintPowerCircuitAt interface offset).asSubcircuit
    constraintPowerName (constraintOffset interface offset)

def pointOp (interface : ShapeInterface shape) (offset : Nat) : Op :=
  .subcircuit (pointSubcircuit interface offset)

def matrixPowerOp (interface : ShapeInterface shape) (offset : Nat) : Op :=
  .subcircuit (matrixPowerSubcircuit interface offset)

def constraintPowerOp (interface : ShapeInterface shape) (offset : Nat) : Op :=
  .subcircuit (constraintPowerSubcircuit interface offset)

def terminalExpr (interface : ShapeInterface shape) (offset : Nat) : KExpr :=
  KExpr.add (interface.eval_K offset) <|
    KExpr.add
      (KExpr.mul (gammaMatrixOutput interface offset)
        (interface.eval_A offset))
      (KExpr.mul (gammaConstraintOutput interface offset) <|
        KExpr.mul (pointEqualityOutput interface offset) <|
          KExpr.add (interface.ccs offset)
            (KExpr.mul (interface.gamma offset) (interface.norm offset)))

def terminalAssertions (interface : ShapeInterface shape) (offset : Nat) : List Expr :=
  KExpr.equalities (interface.terminal offset) (terminalExpr interface offset)

def opsAt (interface : ShapeInterface shape) (offset : Nat) : List Op :=
  [pointOp interface offset, matrixPowerOp interface offset,
    constraintPowerOp interface offset] ++
      (terminalAssertions interface offset).map Op.assertZero

def main (interface : ShapeInterface shape) : Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops (interface : ShapeInterface shape) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

structure Assumptions (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) : Prop where
  point : PointEquality.Owned.Assumptions
    (pointInterfaceAt interface offset) offset env
  gammaBelow : (interface.gamma offset).VarsBelow offset
  eval_KBelow : (interface.eval_K offset).VarsBelow offset
  eval_ABelow : (interface.eval_A offset).VarsBelow offset
  ccsBelow : (interface.ccs offset).VarsBelow offset
  normBelow : (interface.norm offset).VarsBelow offset
  terminalBelow : (interface.terminal offset).VarsBelow offset

def referenceTerminal (interface : ShapeInterface shape) (offset : Nat) (env : Env) : K :=
  K.add (interface.eval_K offset |>.eval env) <|
    K.add
      (K.mul
        (TargetPolynomial.power extensionOps.toOps
          (interface.gamma offset |>.eval env) (matrixExponentOf shape))
        (interface.eval_A offset |>.eval env))
      (K.mul
        (TargetPolynomial.power extensionOps.toOps
          (interface.gamma offset |>.eval env) (constraintExponentOf shape)) <|
        K.mul
          (SumCheckTruthPath.pointEquality extensionOps
            (PointEquality.Owned.evalLeftPoint
              (pointInterfaceAt interface offset) offset env)
            (PointEquality.Owned.evalRightPoint
              (pointInterfaceAt interface offset) offset env)) <|
          K.add (interface.ccs offset |>.eval env)
            (K.mul (interface.gamma offset |>.eval env)
              (interface.norm offset |>.eval env)))

def SpecHolds (interface : ShapeInterface shape) (offset : Nat) (env : Env) : Prop :=
  (interface.terminal offset).eval env = referenceTerminal interface offset env

theorem specHolds_at_iff_of_fields_eq (interface : ShapeInterface shape)
    (leftOffset rightOffset : Nat) (env : Env)
    (roundPointEq : interface.roundPoint leftOffset =
      interface.roundPoint rightOffset)
    (alphaEq : interface.alpha leftOffset = interface.alpha rightOffset)
    (gammaEq : interface.gamma leftOffset = interface.gamma rightOffset)
    (evalKEq : interface.eval_K leftOffset = interface.eval_K rightOffset)
    (evalAEq : interface.eval_A leftOffset = interface.eval_A rightOffset)
    (ccsEq : interface.ccs leftOffset = interface.ccs rightOffset)
    (normEq : interface.norm leftOffset = interface.norm rightOffset)
    (terminalEq : interface.terminal leftOffset =
      interface.terminal rightOffset) :
    SpecHolds interface leftOffset env ↔
      SpecHolds interface rightOffset env := by
  have leftPointEq : PointEquality.Owned.evalLeftPoint
      (pointInterfaceAt interface leftOffset) leftOffset env =
      PointEquality.Owned.evalLeftPoint
        (pointInterfaceAt interface rightOffset) rightOffset env := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices shape.cubeVariables).map
        (fun coordinate =>
          (interface.roundPoint leftOffset coordinate).eval env) =
      (canonicalFinIndices shape.cubeVariables).map
        (fun coordinate =>
          (interface.roundPoint rightOffset coordinate).eval env)
    rw [roundPointEq]
  have rightPointEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface leftOffset) leftOffset env =
      PointEquality.Owned.evalRightPoint
        (pointInterfaceAt interface rightOffset) rightOffset env := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices shape.cubeVariables).map
        (fun coordinate => (interface.alpha leftOffset coordinate).eval env) =
      (canonicalFinIndices shape.cubeVariables).map
        (fun coordinate => (interface.alpha rightOffset coordinate).eval env)
    rw [alphaEq]
  unfold SpecHolds referenceTerminal
  rw [leftPointEq, rightPointEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq,
    terminalEq]

private theorem pointLength_eq_of_positive (interface : ShapeInterface shape) (offset : Nat)
    (positive : 0 < shape.cubeVariables) :
    pointLength interface offset = 4 * shape.cubeVariables - 2 := by
  unfold pointLength pointCircuitAt
  exact PointEquality.Owned.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset positive

private theorem matrixLength_eq_mul (interface : ShapeInterface shape) (offset : Nat) :
    matrixLength interface offset = 3 * matrixExponentOf shape := by
  unfold matrixLength matrixPowerCircuitAt
  rw [Power.localLength_eq]

private theorem constraintLength_eq_mul (interface : ShapeInterface shape) (offset : Nat) :
    constraintLength interface offset = 3 * constraintFactorOf shape := by
  unfold constraintLength constraintPowerCircuitAt
  rw [Power.localLength_eq]

private theorem matrixAssumptionsAt (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) {source : Env}
    (assumptions : Assumptions interface offset source) :
    Power.Assumptions (matrixExponentOf shape)
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env := by
  apply Power.assumptions_of_point_varsBelow
  have offsetLe : offset ≤ matrixOffset interface offset := by
    unfold matrixOffset
    omega
  simpa [matrixPowerInterfaceAt] using
    (interface.gamma offset).varsBelow_mono assumptions.gammaBelow offsetLe

private theorem constraintAssumptionsAt (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) {source : Env}
    (assumptions : Assumptions interface offset source) :
    Power.Assumptions (constraintFactorOf shape)
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env := by
  apply Power.assumptions_of_point_varsBelow
  have below := Power.output_varsBelow (matrixExponentOf shape)
    (matrixPowerInterfaceAt interface offset) (matrixOffset interface offset)
      env (matrixAssumptionsAt interface offset env assumptions)
  have matrixEnd : matrixOffset interface offset + 3 * (matrixExponentOf shape) =
      constraintOffset interface offset := by
    unfold constraintOffset matrixLength matrixPowerCircuitAt
    rw [Power.localLength_eq]
  rw [matrixEnd] at below
  exact below

private theorem pointCall_sound (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env := by
  have callHolds := rows (pointOp interface offset) (by simp [opsAt])
  change (pointCircuitAt interface offset).assumptions offset env →
    (pointCircuitAt interface offset).spec offset env at callHolds
  exact callHolds assumptions.point

private theorem matrixPowerCall_sound (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Power.SpecHolds (matrixExponentOf shape) (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env := by
  have callHolds := rows (matrixPowerOp interface offset) (by simp [opsAt])
  change (matrixPowerCircuitAt interface offset).assumptions
      (matrixOffset interface offset) env →
    (matrixPowerCircuitAt interface offset).spec
      (matrixOffset interface offset) env at callHolds
  exact callHolds (matrixAssumptionsAt interface offset env assumptions)

private theorem constraintPowerCall_sound (interface : ShapeInterface shape)
    (offset : Nat) (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Power.SpecHolds (constraintFactorOf shape)
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env := by
  have callHolds := rows (constraintPowerOp interface offset) (by simp [opsAt])
  change (constraintPowerCircuitAt interface offset).assumptions
      (constraintOffset interface offset) env →
    (constraintPowerCircuitAt interface offset).spec
      (constraintOffset interface offset) env at callHolds
  exact callHolds (constraintAssumptionsAt interface offset env assumptions)

private theorem terminalRows_sound (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset)) :
    (interface.terminal offset).eval env =
      (terminalExpr interface offset).eval env := by
  apply (KExpr.equalities_hold_iff env (interface.terminal offset)
    (terminalExpr interface offset)).mp
  intro expression member
  exact rows (Op.assertZero expression) (by
    simp [opsAt, terminalAssertions, member])

private theorem terminalExpr_eval_of_children (interface : ShapeInterface shape)
    (offset : Nat) (env : Env)
    (pointSpec : PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env)
    (matrixSpec : Power.SpecHolds (matrixExponentOf shape)
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env)
    (constraintSpec : Power.SpecHolds (constraintFactorOf shape)
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env) :
    (terminalExpr interface offset).eval env =
      referenceTerminal interface offset env := by
  have matrixEq := Power.spec_implies_power (matrixExponentOf shape)
    (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env matrixSpec
  have constraintEq := Power.spec_implies_power (constraintFactorOf shape)
    (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env constraintSpec
  change (gammaMatrixOutput interface offset).eval env =
    TargetPolynomial.power extensionOps.toOps
      ((interface.gamma offset).eval env) (matrixExponentOf shape) at matrixEq
  change (gammaConstraintOutput interface offset).eval env =
    TargetPolynomial.power extensionOps.toOps
      ((gammaMatrixOutput interface offset).eval env) (constraintFactorOf shape)
    at constraintEq
  rw [matrixEq, ← power_mul, ← constraintExponent_eq_mul shape] at constraintEq
  unfold gammaMatrixOutput at matrixEq
  unfold gammaConstraintOutput at constraintEq
  unfold PointEquality.Owned.SpecHolds at pointSpec
  unfold terminalExpr referenceTerminal pointEqualityOutput gammaMatrixOutput
    gammaConstraintOutput
  simp only [KExpr.eval_add, KExpr.eval_mul]
  rw [pointSpec, matrixEq, constraintEq]

theorem soundness (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  have pointSpec := pointCall_sound interface offset env rows assumptions
  have matrixSpec := matrixPowerCall_sound interface offset env rows assumptions
  have constraintSpec := constraintPowerCall_sound interface offset env rows
    assumptions
  exact (terminalRows_sound interface offset env rows).trans
    (terminalExpr_eval_of_children interface offset env pointSpec matrixSpec
      constraintSpec)

private theorem evalLeftPoint_eq_of_agree_below (interface : ShapeInterface shape)
    (offset : Nat) (before after : Env)
    (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index) :
    PointEquality.Owned.evalLeftPoint (pointInterfaceAt interface offset)
        offset after =
      PointEquality.Owned.evalLeftPoint (pointInterfaceAt interface offset)
        offset before := by
  apply cubePoint_eq_of_coordinates
  change (canonicalFinIndices shape.cubeVariables).map
      (fun coordinate => (interface.roundPoint offset coordinate).eval after) =
    (canonicalFinIndices shape.cubeVariables).map
      (fun coordinate => (interface.roundPoint offset coordinate).eval before)
  apply List.map_congr_left
  intro coordinate _
  exact (interface.roundPoint offset coordinate).eval_eq_of_agree_below
    offset after before (assumptions.point coordinate).1 agrees

private theorem evalRightPoint_eq_of_agree_below (interface : ShapeInterface shape)
    (offset : Nat) (before after : Env)
    (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index) :
    PointEquality.Owned.evalRightPoint (pointInterfaceAt interface offset)
        offset after =
      PointEquality.Owned.evalRightPoint (pointInterfaceAt interface offset)
        offset before := by
  apply cubePoint_eq_of_coordinates
  change (canonicalFinIndices shape.cubeVariables).map
      (fun coordinate => (interface.alpha offset coordinate).eval after) =
    (canonicalFinIndices shape.cubeVariables).map
      (fun coordinate => (interface.alpha offset coordinate).eval before)
  apply List.map_congr_left
  intro coordinate _
  exact (interface.alpha offset coordinate).eval_eq_of_agree_below
    offset after before (assumptions.point coordinate).2 agrees

theorem specHolds_of_agree_below (interface : ShapeInterface shape) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have terminalEq := (interface.terminal offset).eval_eq_of_agree_below
    offset after before assumptions.terminalBelow agrees
  have gammaEq := (interface.gamma offset).eval_eq_of_agree_below
    offset after before assumptions.gammaBelow agrees
  have evalKEq := (interface.eval_K offset).eval_eq_of_agree_below
    offset after before assumptions.eval_KBelow agrees
  have evalAEq := (interface.eval_A offset).eval_eq_of_agree_below
    offset after before assumptions.eval_ABelow agrees
  have ccsEq := (interface.ccs offset).eval_eq_of_agree_below
    offset after before assumptions.ccsBelow agrees
  have normEq := (interface.norm offset).eval_eq_of_agree_below
    offset after before assumptions.normBelow agrees
  have leftEq := evalLeftPoint_eq_of_agree_below interface offset before after
    assumptions agrees
  have rightEq := evalRightPoint_eq_of_agree_below interface offset before after
    assumptions agrees
  unfold SpecHolds referenceTerminal at specification ⊢
  rw [terminalEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq, leftEq, rightEq]
  exact specification

private theorem pointOp_flatConstraints (interface : ShapeInterface shape) (offset : Nat) :
    (pointOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main
        offset) := by
  unfold pointOp pointSubcircuit
  exact FormalCircuit.asSubcircuit_constraints _ _ _

private theorem matrixPowerOp_flatConstraints (interface : ShapeInterface shape)
    (offset : Nat) :
    (matrixPowerOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset)) := by
  unfold matrixPowerOp matrixPowerSubcircuit
  exact FormalCircuit.asSubcircuit_constraints _ _ _

private theorem constraintPowerOp_flatConstraints (interface : ShapeInterface shape)
    (offset : Nat) :
    (constraintPowerOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops
        (constraintPowerCircuitAt interface offset).main
        (constraintOffset interface offset)) := by
  unfold constraintPowerOp constraintPowerSubcircuit
  exact FormalCircuit.asSubcircuit_constraints _ _ _

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest ih =>
      change [expression] ++ flatConstraints (rest.map Op.assertZero) =
        expression :: rest
      rw [ih]
      rfl

theorem flatConstraints_opsAt (interface : ShapeInterface shape) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset) ++
      flatConstraints (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset)) ++
      flatConstraints (Circuit.ops
        (constraintPowerCircuitAt interface offset).main
        (constraintOffset interface offset)) ++
      terminalAssertions interface offset := by
  unfold opsAt
  rw [flatConstraints_append, flatConstraints_assertions]
  simp only [flatConstraints, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, pointOp_flatConstraints,
    matrixPowerOp_flatConstraints, constraintPowerOp_flatConstraints]
  simp only [List.append_assoc]

private theorem pointOp_localLength (interface : ShapeInterface shape) (offset : Nat) :
    (pointOp interface offset).localLength = pointLength interface offset := by
  rfl

private theorem matrixPowerOp_localLength (interface : ShapeInterface shape)
    (offset : Nat) :
    (matrixPowerOp interface offset).localLength =
      matrixLength interface offset := by
  rfl

private theorem constraintPowerOp_localLength (interface : ShapeInterface shape)
    (offset : Nat) :
    (constraintPowerOp interface offset).localLength =
      constraintLength interface offset := by
  rfl

private theorem assertions_localLength (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest ih =>
      simp only [List.map_cons, localLength, Op.localLength, List.sum_cons,
        Nat.zero_add]
      simpa [localLength] using ih

private theorem localLength_append (left right : List Op) :
    localLength (left ++ right) = localLength left + localLength right := by
  unfold localLength
  rw [List.map_append, List.sum_append]

private theorem localLength_three (first second third : Op) :
    localLength [first, second, third] =
      first.localLength + (second.localLength + third.localLength) := by
  rfl

private theorem opsAt_localLength (interface : ShapeInterface shape) (offset : Nat) :
    localLength (opsAt interface offset) = pointLength interface offset +
      matrixLength interface offset + constraintLength interface offset := by
  calc
    localLength (opsAt interface offset) =
        localLength [pointOp interface offset, matrixPowerOp interface offset,
          constraintPowerOp interface offset] +
            localLength ((terminalAssertions interface offset).map
              Op.assertZero) := by
      rw [opsAt, localLength_append]
    _ = (pointOp interface offset).localLength +
          ((matrixPowerOp interface offset).localLength +
            (constraintPowerOp interface offset).localLength) := by
      rw [localLength_three, assertions_localLength, Nat.add_zero]
    _ = pointLength interface offset + matrixLength interface offset +
          constraintLength interface offset := by
      rw [pointOp_localLength, matrixPowerOp_localLength,
        constraintPowerOp_localLength]
      omega

private theorem pointRows_preservedAfterMatrix (interface : ShapeInterface shape)
    (offset : Nat) (env afterPoint afterMatrix : Env)
    (assumptions : Assumptions interface offset env) (positive : 0 < shape.cubeVariables)
    (pointRows : holdsFlat afterPoint
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (matrixAgrees : AgreesOutside afterPoint afterMatrix
      (matrixOffset interface offset) (matrixLength interface offset)) :
    holdsFlat afterMatrix
      (Circuit.ops (pointCircuitAt interface offset).main offset) := by
  unfold holdsFlat at pointRows ⊢
  apply constraintsHold_of_agree_below afterPoint afterMatrix _
    (matrixOffset interface offset)
  · have scope := PointEquality.Owned.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
    intro expression member
    have below := scope expression member
    apply Expr.VarsBelow.mono expression below
    rw [PointEquality.Owned.program_recipes_length_of_positive
      (pointInterfaceAt interface offset) offset positive]
    unfold matrixOffset
    rw [pointLength_eq_of_positive interface offset positive]
  · intro index below
    exact matrixAgrees index (Or.inl below)
  · exact pointRows

private theorem pointRows_preservedAfterConstraint (interface : ShapeInterface shape)
    (offset : Nat) (env afterMatrix completed : Env)
    (assumptions : Assumptions interface offset env) (positive : 0 < shape.cubeVariables)
    (pointRows : holdsFlat afterMatrix
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (constraintAgrees : AgreesOutside afterMatrix completed
      (constraintOffset interface offset) (constraintLength interface offset)) :
    holdsFlat completed
      (Circuit.ops (pointCircuitAt interface offset).main offset) := by
  unfold holdsFlat at pointRows ⊢
  apply constraintsHold_of_agree_below afterMatrix completed _
    (constraintOffset interface offset)
  · have scope := PointEquality.Owned.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
    intro expression member
    apply Expr.VarsBelow.mono expression (scope expression member)
    rw [PointEquality.Owned.program_recipes_length_of_positive
      (pointInterfaceAt interface offset) offset positive]
    unfold constraintOffset matrixOffset
    rw [pointLength_eq_of_positive interface offset positive, matrixLength_eq_mul]
    omega
  · intro index below
    exact constraintAgrees index (Or.inl below)
  · exact pointRows

private theorem matrixRows_preservedAfterConstraint (interface : ShapeInterface shape)
    (offset : Nat) (env afterMatrix completed : Env)
    (assumptions : Assumptions interface offset env)
    (matrixRows : holdsFlat afterMatrix
      (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset)))
    (constraintAgrees : AgreesOutside afterMatrix completed
      (constraintOffset interface offset) (constraintLength interface offset)) :
    holdsFlat completed (Circuit.ops (matrixPowerCircuitAt interface offset).main
      (matrixOffset interface offset)) := by
  unfold holdsFlat at matrixRows ⊢
  apply constraintsHold_of_agree_below afterMatrix completed _
    (constraintOffset interface offset)
  · have scope := Power.flatConstraints_varsBelow_exact (matrixExponentOf shape)
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env
      (matrixAssumptionsAt interface offset env assumptions)
    intro expression member
    apply Expr.VarsBelow.mono expression (scope expression member)
    unfold constraintOffset
    rw [matrixLength_eq_mul]
  · intro index below
    exact constraintAgrees index (Or.inl below)
  · exact matrixRows

private theorem terminalRows_complete (interface : ShapeInterface shape) (offset : Nat)
    (env completed : Env) (assumptions : Assumptions interface offset env)
    (agrees : AgreesOutside env completed offset
      (pointLength interface offset + matrixLength interface offset +
        constraintLength interface offset))
    (specification : SpecHolds interface offset env)
    (pointRows : holdsFlat completed
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (matrixRows : holdsFlat completed
      (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset)))
    (constraintRows : holdsFlat completed
      (Circuit.ops (constraintPowerCircuitAt interface offset).main
        (constraintOffset interface offset))) :
    ConstraintsHold completed (terminalAssertions interface offset) := by
  have belowAgrees : ∀ index, index < offset →
      completed index = env index := fun index below =>
    agrees index (Or.inl below)
  have semantic := specHolds_of_agree_below interface offset env completed
    assumptions belowAgrees specification
  have pointSpec := PointEquality.Owned.soundness
    (pointInterfaceAt interface offset) completed offset assumptions.point
      (holdsFlat_implies_holds completed _ pointRows)
  have matrixSpec := Power.soundness (matrixExponentOf shape)
    (matrixPowerInterfaceAt interface offset) completed
      (matrixOffset interface offset)
      (matrixAssumptionsAt interface offset completed assumptions)
      (holdsFlat_implies_holds completed _ matrixRows)
  have constraintSpec := Power.soundness (constraintFactorOf shape)
    (constraintPowerInterfaceAt interface offset) completed
      (constraintOffset interface offset)
      (constraintAssumptionsAt interface offset completed assumptions)
      (holdsFlat_implies_holds completed _ constraintRows)
  apply (KExpr.equalities_hold_iff completed (interface.terminal offset)
    (terminalExpr interface offset)).mpr
  exact semantic.trans
    (terminalExpr_eval_of_children interface offset completed pointSpec
      matrixSpec constraintSpec).symm

/-- Completeness for any shape with a cube variable. -/
theorem shapeCompleteness (interface : ShapeInterface shape)
    (positive : 0 < shape.cubeVariables) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases PointEquality.Owned.build (pointInterfaceAt interface offset) env
      offset assumptions.point with ⟨afterPoint, pointAgrees, pointRows⟩
  rcases Power.build (matrixExponentOf shape) (matrixPowerInterfaceAt interface offset)
      afterPoint (matrixOffset interface offset)
      (matrixAssumptionsAt interface offset afterPoint assumptions) with
    ⟨afterMatrix, matrixAgrees, matrixRows⟩
  have pointMatrixAgrees : AgreesOutside env afterMatrix offset
      (pointLength interface offset + matrixLength interface offset) := by
    have matrixAgreesAt : AgreesOutside afterPoint afterMatrix
        (offset + pointLength interface offset)
        (matrixLength interface offset) := by
      simpa only [matrixOffset] using! matrixAgrees
    exact pointAgrees.append matrixAgreesAt
  rcases Power.build (constraintFactorOf shape)
      (constraintPowerInterfaceAt interface offset) afterMatrix
      (constraintOffset interface offset)
      (constraintAssumptionsAt interface offset afterMatrix assumptions) with
    ⟨completed, constraintAgrees, constraintRows⟩
  have combinedAgrees : AgreesOutside env completed offset
      (pointLength interface offset + matrixLength interface offset +
        constraintLength interface offset) := by
    have constraintAgreesAt : AgreesOutside afterMatrix completed
        (offset + (pointLength interface offset + matrixLength interface offset))
        (constraintLength interface offset) := by
      rw [← constraintOffset_eq]
      exact constraintAgrees
    exact pointMatrixAgrees.append constraintAgreesAt
  have pointRowsAfterMatrix := pointRows_preservedAfterMatrix interface offset
    env afterPoint afterMatrix assumptions positive pointRows matrixAgrees
  have pointRowsCompleted := pointRows_preservedAfterConstraint interface offset
    env afterMatrix completed assumptions positive pointRowsAfterMatrix constraintAgrees
  have matrixRowsCompleted := matrixRows_preservedAfterConstraint interface
    offset env afterMatrix completed assumptions matrixRows constraintAgrees
  have terminalRows := terminalRows_complete interface offset env completed
    assumptions combinedAgrees specification pointRowsCompleted
      matrixRowsCompleted constraintRows
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset
      (localLength (opsAt interface offset))
    rw [opsAt_localLength]
    exact combinedAgrees
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨(constraintsHold_append completed _ _).mpr
      ⟨(constraintsHold_append completed _ _).mpr
          ⟨pointRowsCompleted, matrixRowsCompleted⟩, constraintRows⟩,
        terminalRows⟩

/-- The final identity circuit for any shape with a cube variable. -/
def shapeCircuit (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables) :
    FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := shapeCompleteness interface positive

private theorem cubeVariables_positive : 0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  shapeCompleteness interface cubeVariables_positive env offset assumptions specification

theorem build (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) :=
  completeness interface env offset assumptions specification

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

private theorem pointLength_eq (interface : Interface) (offset : Nat) :
    pointLength interface offset = 110 := by
  rw [pointLength_eq_of_positive interface offset cubeVariables_positive]
  rfl

private theorem matrixLength_eq (interface : Interface) (offset : Nat) :
    matrixLength interface offset = 2592 := by
  rw [matrixLength_eq_mul, matrixExponent_eq]

private theorem constraintLength_eq (interface : Interface) (offset : Nat) :
    constraintLength interface offset = 15 := by
  rw [constraintLength_eq_mul, constraintFactor_eq]

private theorem shapePointOutput_varsBelow (interface : ShapeInterface shape)
    (positive : 0 < shape.cubeVariables) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    (pointEqualityOutput interface offset).VarsBelow
      (matrixOffset interface offset) := by
  have below := PointEquality.Owned.output_varsBelow
    (pointInterfaceAt interface offset) offset env assumptions.point
  unfold pointEqualityOutput
  apply KExpr.varsBelow_mono _ below
  rw [PointEquality.Owned.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset positive]
  unfold matrixOffset
  rw [pointLength_eq_of_positive interface offset positive]

private theorem shapeMatrixOutput_varsBelow (interface : ShapeInterface shape)
    (offset : Nat) (env : Env) (assumptions : Assumptions interface offset env) :
    (gammaMatrixOutput interface offset).VarsBelow
      (constraintOffset interface offset) := by
  have below := Power.output_varsBelow (matrixExponentOf shape)
    (matrixPowerInterfaceAt interface offset) (matrixOffset interface offset)
      env (matrixAssumptionsAt interface offset env assumptions)
  apply KExpr.varsBelow_mono _ below
  unfold constraintOffset
  rw [matrixLength_eq_mul]

private theorem shapeConstraintOutput_varsBelow (interface : ShapeInterface shape)
    (offset : Nat) (env : Env) (assumptions : Assumptions interface offset env) :
    (gammaConstraintOutput interface offset).VarsBelow
      (finalOffset interface offset) := by
  have below := Power.output_varsBelow (constraintFactorOf shape)
    (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env
      (constraintAssumptionsAt interface offset env assumptions)
  apply KExpr.varsBelow_mono _ below
  unfold finalOffset
  rw [constraintLength_eq_mul]

private theorem shapeTerminalExpr_varsBelow (interface : ShapeInterface shape)
    (positive : 0 < shape.cubeVariables) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    (terminalExpr interface offset).VarsBelow (finalOffset interface offset) := by
  have offsetLe : offset ≤ finalOffset interface offset := by
    unfold finalOffset constraintOffset matrixOffset
    omega
  unfold terminalExpr
  apply KExpr.add_varsBelow
  · exact (interface.eval_K offset).varsBelow_mono assumptions.eval_KBelow
      offsetLe
  · apply KExpr.add_varsBelow
    · exact KExpr.mul_varsBelow _ _ _
        (KExpr.varsBelow_mono _
          (shapeMatrixOutput_varsBelow interface offset env assumptions)
          (by unfold finalOffset; omega))
        ((interface.eval_A offset).varsBelow_mono assumptions.eval_ABelow
          offsetLe)
    · apply KExpr.mul_varsBelow
      · exact shapeConstraintOutput_varsBelow interface offset env assumptions
      · apply KExpr.mul_varsBelow
        · exact KExpr.varsBelow_mono _
            (shapePointOutput_varsBelow interface positive offset env assumptions) (by
              unfold finalOffset constraintOffset matrixOffset
              omega)
        · apply KExpr.add_varsBelow
          · exact (interface.ccs offset).varsBelow_mono assumptions.ccsBelow
              offsetLe
          · exact KExpr.mul_varsBelow _ _ _
              ((interface.gamma offset).varsBelow_mono assumptions.gammaBelow
                offsetLe)
              ((interface.norm offset).varsBelow_mono assumptions.normBelow
                offsetLe)

/-- Every row of the final identity at any shape reads only words below its end. -/
theorem shapeFlatConstraints_varsBelow (interface : ShapeInterface shape)
    (positive : 0 < shape.cubeVariables) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
  have pointLengthEq := pointLength_eq_of_positive interface offset positive
  have matrixLengthEq := matrixLength_eq_mul interface offset
  have constraintLengthEq := constraintLength_eq_mul interface offset
  intro expression member
  rcases List.mem_append.mp member with coreMember | terminalMember
  · rcases List.mem_append.mp coreMember with firstTwoMember |
        constraintMember
    · rcases List.mem_append.mp firstTwoMember with pointMember |
          matrixMember
      · have below := PointEquality.Owned.flatConstraints_varsBelow
          (pointInterfaceAt interface offset) offset env assumptions.point
            expression pointMember
        apply Expr.VarsBelow.mono expression below
        rw [PointEquality.Owned.program_recipes_length_of_positive
          (pointInterfaceAt interface offset) offset positive]
        omega
      · have below := Power.flatConstraints_varsBelow_exact (matrixExponentOf shape)
          (matrixPowerInterfaceAt interface offset)
            (matrixOffset interface offset) env
            (matrixAssumptionsAt interface offset env assumptions)
            expression matrixMember
        apply Expr.VarsBelow.mono expression below
        unfold matrixOffset
        omega
    · have below := Power.flatConstraints_varsBelow_exact (constraintFactorOf shape)
        (constraintPowerInterfaceAt interface offset)
          (constraintOffset interface offset) env
          (constraintAssumptionsAt interface offset env assumptions)
          expression constraintMember
      apply Expr.VarsBelow.mono expression below
      unfold constraintOffset matrixOffset
      omega
  · exact KExpr.equalities_varsBelow (interface.terminal offset)
      (terminalExpr interface offset)
      (offset + (pointLength interface offset + matrixLength interface offset +
        constraintLength interface offset))
      ((interface.terminal offset).varsBelow_mono assumptions.terminalBelow
        (by omega))
      (by
        have below := shapeTerminalExpr_varsBelow interface positive offset env assumptions
        apply KExpr.varsBelow_mono _ below
        unfold finalOffset constraintOffset matrixOffset
        omega)
      expression terminalMember

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  shapeFlatConstraints_varsBelow interface cubeVariables_positive offset env assumptions

def privateCount : Nat := 2717

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 2717 := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength, pointLength_eq, matrixLength_eq, constraintLength_eq]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 5 := by
  change (opsAt interface offset).length = 5
  simp [opsAt, terminalAssertions, KExpr.equalities]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      2719 := by
  change (flatConstraints (opsAt interface offset)).length = _
  have pointFlat :
      (flatConstraints (Circuit.ops (pointCircuitAt interface offset).main
        offset)).length = 110 := by
    unfold pointCircuitAt
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      PointEquality.Owned.flatConstraints_length_of_positive
        (pointInterfaceAt interface offset) offset (by
          norm_num [productionShape, Phi81MatrixSource.phi81Shape,
            cubeVariables])
  have matrixFlat :
      (flatConstraints (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset))).length = 2592 := by
    unfold matrixPowerCircuitAt
    rw [Power.flatConstraints_length, matrixExponent_eq]
  have constraintFlat :
      (flatConstraints (Circuit.ops
        (constraintPowerCircuitAt interface offset).main
        (constraintOffset interface offset))).length = 15 := by
    unfold constraintPowerCircuitAt
    rw [Power.flatConstraints_length, constraintFactor_eq]
  rw [flatConstraints_opsAt, List.length_append, List.length_append,
    List.length_append, pointFlat, matrixFlat, constraintFlat]
  simp [terminalAssertions, KExpr.equalities]

/-- The leaf's terminal is `terminalFromMessage` of SuperNeo v1.2 Section 7.3, Step 4, for any
verifier input, challenges and output message of a shape with one fresh source. -/
theorem spec_implies_terminal (interface : ShapeInterface shape) (offset : Nat) (env : Env)
    (freshOne : shape.freshCount = 1) (input : ProtocolPolynomial.VerifierInput K shape)
    (alpha roundPoint : CubePoint K shape.cubeVariables) (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K shape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (pointInterfaceAt interface offset) offset env = roundPoint)
    (alphaEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface offset) offset env = alpha)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps input gamma roundPoint message)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps input gamma roundPoint message)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps input gamma message)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message)
    (specification : SpecHolds interface offset env) :
    (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps input alpha gamma roundPoint
        message := by
  unfold SpecHolds referenceTerminal at specification
  rw [roundPointEq, alphaEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq] at specification
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_of_one freshOne]
  simpa [matrixExponentOf, constraintExponentOf] using! specification

/-- The converse of `spec_implies_terminal`: when the terminal word carries the verifier's
terminal value (SuperNeo v1.2 Section 7.3, Step 4), the final identity holds. -/
theorem terminal_implies_spec (interface : ShapeInterface shape) (offset : Nat) (env : Env)
    (freshOne : shape.freshCount = 1) (input : ProtocolPolynomial.VerifierInput K shape)
    (alpha roundPoint : CubePoint K shape.cubeVariables) (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K shape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (pointInterfaceAt interface offset) offset env = roundPoint)
    (alphaEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface offset) offset env = alpha)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps input gamma roundPoint message)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps input gamma roundPoint message)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps input gamma message)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message)
    (terminalEq : (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps input alpha gamma roundPoint
        message) :
    SpecHolds interface offset env := by
  unfold SpecHolds referenceTerminal
  rw [roundPointEq, alphaEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq, terminalEq]
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_of_one freshOne]
  simp [extensionOps, matrixExponentOf, constraintExponentOf]

theorem spec_implies_keyTerminal
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
    (alphaEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface offset) offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha)
    (gammaEq : (interface.gamma offset).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (specification : SpecHolds interface offset env) :
    (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
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
  unfold SpecHolds referenceTerminal at specification
  rw [roundPointEq, alphaEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq]
    at specification
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_eq]
  simpa [input, execution, message, matrixExponentOf, constraintExponentOf] using!
    specification

theorem keyTerminal_implies_spec
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
    (alphaEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface offset) offset env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha)
    (gammaEq : (interface.gamma offset).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (terminalEq : (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output) :
    SpecHolds interface offset env := by
  let input := (ChallengeDerivation.productionContext
    relation ajtai running fresh).input
  let execution := (ProductionKey.key relation ajtai).piCcsExecution
    running fresh proof
  let message := ((ProductionKey.key relation ajtai).piCcsCertificate
    running fresh proof).output
  unfold SpecHolds referenceTerminal
  rw [roundPointEq, alphaEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq]
  rw [terminalEq]
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_eq]
  simp [extensionOps, matrixExponentOf, constraintExponentOf]

end Shape

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity
