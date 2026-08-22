import NightstreamFPrime.Gadgets.Multilinear.PointEquality
import NightstreamFPrime.Gadgets.Polynomial.Power
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.v1_1.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 4, `v = Q(r')`.
Obligation: Enforce
`E_K + gamma^864 E_A + gamma^12960 eq(r',alpha) (F + gamma N)`.

Inputs:
- verifier-derived `r'`, `alpha`, and `gamma`;
- the already-constrained `E_K`, `E_A`, `F`, and `N` leaf outputs;
- the SumCheck terminal claim `v`.

Outputs:
- the exact complete v1.1 PiCCS terminal equality.

Constraint groups:
- C1: one opaque owned `PointEquality` child;
- C2: one opaque owned `Power` child for `gamma^864`;
- C3: one opaque owned `Power` child for `gamma^12960`;
- C4: two extension-component final-identity assertions.

Parent coverage:
- `ProtocolPolynomial.terminalFromMessage` in `PiCCS.v1_1.Coverage.chain`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity

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

def matrixExponent : Nat := productionShape.matrixEvaluationOffset
def constraintExponent : Nat := productionShape.constraintOffset

theorem matrixExponent_eq : matrixExponent = 864 := by
  norm_num [matrixExponent, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.matrixEvaluationOffset,
    Shape.padEvaluationCount, ringDegree]

theorem constraintExponent_eq : constraintExponent = 12960 := by
  norm_num [constraintExponent, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.constraintOffset,
    Shape.padEvaluationCount, Shape.matrixEvaluationCount, ringDegree]

theorem gammaFreshPower_eq (gamma : K) :
    TargetPolynomial.power extensionOps.toOps gamma
      productionShape.freshCount = gamma := by
  simpa [productionShape, productionProfile, Phi81MatrixSource.phi81Shape,
    TargetPolynomial.power] using extensionLaws.mul_one gamma

structure Interface where
  roundPoint : Nat → Fin productionShape.cubeVariables → KExpr
  alpha : Nat → Fin productionShape.cubeVariables → KExpr
  gamma : Nat → KExpr
  eval_K : Nat → KExpr
  eval_A : Nat → KExpr
  ccs : Nat → KExpr
  norm : Nat → KExpr
  terminal : Nat → KExpr

def pointInterfaceAt (interface : Interface) (parentOffset : Nat) :
    PointEquality.Owned.Interface productionShape.cubeVariables where
  left := fun _ => interface.roundPoint parentOffset
  right := fun _ => interface.alpha parentOffset

def matrixPowerInterfaceAt (interface : Interface) (parentOffset : Nat) :
    Power.Interface where
  point := fun _ => interface.gamma parentOffset

def constraintPowerInterfaceAt (interface : Interface) (parentOffset : Nat) :
    Power.Interface where
  point := fun _ => interface.gamma parentOffset

def pointCircuitAt (interface : Interface) (parentOffset : Nat) :
    FormalCircuit :=
  PointEquality.Owned.circuit (pointInterfaceAt interface parentOffset)

def matrixPowerCircuitAt (interface : Interface) (parentOffset : Nat) :
    FormalCircuit :=
  Power.circuit matrixExponent (matrixPowerInterfaceAt interface parentOffset)

def constraintPowerCircuitAt (interface : Interface) (parentOffset : Nat) :
    FormalCircuit :=
  Power.circuit constraintExponent
    (constraintPowerInterfaceAt interface parentOffset)

def pointLength (interface : Interface) (offset : Nat) : Nat :=
  localLength (Circuit.ops (pointCircuitAt interface offset).main offset)

def matrixOffset (interface : Interface) (offset : Nat) : Nat :=
  offset + pointLength interface offset

def matrixLength (interface : Interface) (offset : Nat) : Nat :=
  localLength (Circuit.ops (matrixPowerCircuitAt interface offset).main
    (matrixOffset interface offset))

def constraintOffset (interface : Interface) (offset : Nat) : Nat :=
  matrixOffset interface offset + matrixLength interface offset

private theorem constraintOffset_eq (interface : Interface) (offset : Nat) :
    constraintOffset interface offset =
      offset + (pointLength interface offset + matrixLength interface offset) := by
  unfold constraintOffset matrixOffset
  exact Nat.add_assoc _ _ _

def constraintLength (interface : Interface) (offset : Nat) : Nat :=
  localLength (Circuit.ops (constraintPowerCircuitAt interface offset).main
    (constraintOffset interface offset))

def finalOffset (interface : Interface) (offset : Nat) : Nat :=
  constraintOffset interface offset + constraintLength interface offset

def pointEqualityOutput (interface : Interface) (offset : Nat) : KExpr :=
  PointEquality.Owned.output (pointInterfaceAt interface offset) offset

def gammaMatrixOutput (interface : Interface) (offset : Nat) : KExpr :=
  Power.output matrixExponent (matrixPowerInterfaceAt interface offset)
    (matrixOffset interface offset)

def gammaConstraintOutput (interface : Interface) (offset : Nat) : KExpr :=
  Power.output constraintExponent (constraintPowerInterfaceAt interface offset)
    (constraintOffset interface offset)

def pointName : String := "piccs.v1_1.final.point_equality"
def matrixPowerName : String := "piccs.v1_1.final.gamma_matrix_offset"
def constraintPowerName : String := "piccs.v1_1.final.gamma_constraint_offset"

def pointSubcircuit (interface : Interface) (offset : Nat) : Subcircuit :=
  (pointCircuitAt interface offset).asSubcircuit pointName offset

def matrixPowerSubcircuit (interface : Interface) (offset : Nat) : Subcircuit :=
  (matrixPowerCircuitAt interface offset).asSubcircuit matrixPowerName
    (matrixOffset interface offset)

def constraintPowerSubcircuit (interface : Interface)
    (offset : Nat) : Subcircuit :=
  (constraintPowerCircuitAt interface offset).asSubcircuit
    constraintPowerName (constraintOffset interface offset)

def pointOp (interface : Interface) (offset : Nat) : Op :=
  .subcircuit (pointSubcircuit interface offset)

def matrixPowerOp (interface : Interface) (offset : Nat) : Op :=
  .subcircuit (matrixPowerSubcircuit interface offset)

def constraintPowerOp (interface : Interface) (offset : Nat) : Op :=
  .subcircuit (constraintPowerSubcircuit interface offset)

def terminalExpr (interface : Interface) (offset : Nat) : KExpr :=
  KExpr.add (interface.eval_K offset) <|
    KExpr.add
      (KExpr.mul (gammaMatrixOutput interface offset)
        (interface.eval_A offset))
      (KExpr.mul (gammaConstraintOutput interface offset) <|
        KExpr.mul (pointEqualityOutput interface offset) <|
          KExpr.add (interface.ccs offset)
            (KExpr.mul (interface.gamma offset) (interface.norm offset)))

def terminalAssertions (interface : Interface) (offset : Nat) : List Expr :=
  KExpr.equalities (interface.terminal offset) (terminalExpr interface offset)

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  [pointOp interface offset, matrixPowerOp interface offset,
    constraintPowerOp interface offset] ++
      (terminalAssertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

structure Assumptions (interface : Interface) (offset : Nat)
    (env : Env) : Prop where
  point : PointEquality.Owned.Assumptions
    (pointInterfaceAt interface offset) offset env
  gammaBelow : (interface.gamma offset).VarsBelow offset
  eval_KBelow : (interface.eval_K offset).VarsBelow offset
  eval_ABelow : (interface.eval_A offset).VarsBelow offset
  ccsBelow : (interface.ccs offset).VarsBelow offset
  normBelow : (interface.norm offset).VarsBelow offset
  terminalBelow : (interface.terminal offset).VarsBelow offset

/-- The exact paper value, expressed only through authoritative entry wires. -/
def referenceTerminal (interface : Interface) (offset : Nat) (env : Env) : K :=
  K.add (interface.eval_K offset |>.eval env) <|
    K.add
      (K.mul
        (TargetPolynomial.power extensionOps.toOps
          (interface.gamma offset |>.eval env) matrixExponent)
        (interface.eval_A offset |>.eval env))
      (K.mul
        (TargetPolynomial.power extensionOps.toOps
          (interface.gamma offset |>.eval env) constraintExponent) <|
        K.mul
          (SumCheckTruthPath.pointEquality extensionOps
            (PointEquality.Owned.evalLeftPoint
              (pointInterfaceAt interface offset) offset env)
            (PointEquality.Owned.evalRightPoint
              (pointInterfaceAt interface offset) offset env)) <|
          K.add (interface.ccs offset |>.eval env)
            (K.mul (interface.gamma offset |>.eval env)
              (interface.norm offset |>.eval env)))

/-- Named semantic predicate: the SumCheck terminal equals exact v1.1
`Q(r')`. Internal child outputs are not caller premises. -/
def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (interface.terminal offset).eval env = referenceTerminal interface offset env

theorem specHolds_at_iff_of_fields_eq (interface : Interface)
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
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun coordinate =>
          (interface.roundPoint leftOffset coordinate).eval env) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun coordinate =>
          (interface.roundPoint rightOffset coordinate).eval env)
    rw [roundPointEq]
  have rightPointEq : PointEquality.Owned.evalRightPoint
      (pointInterfaceAt interface leftOffset) leftOffset env =
      PointEquality.Owned.evalRightPoint
        (pointInterfaceAt interface rightOffset) rightOffset env := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun coordinate => (interface.alpha leftOffset coordinate).eval env) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun coordinate => (interface.alpha rightOffset coordinate).eval env)
    rw [alphaEq]
  unfold SpecHolds referenceTerminal
  rw [leftPointEq, rightPointEq, gammaEq, evalKEq, evalAEq, ccsEq, normEq,
    terminalEq]

private theorem pointLength_eq (interface : Interface) (offset : Nat) :
    pointLength interface offset = 94 := by
  unfold pointLength pointCircuitAt
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
    PointEquality.Owned.localLength_eq_of_positive
      (pointInterfaceAt interface offset) offset (by
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables])

private theorem matrixLength_eq (interface : Interface) (offset : Nat) :
    matrixLength interface offset = 1728 := by
  unfold matrixLength matrixPowerCircuitAt
  rw [Power.localLength_eq, matrixExponent_eq]

private theorem constraintLength_eq (interface : Interface) (offset : Nat) :
    constraintLength interface offset = 25920 := by
  unfold constraintLength constraintPowerCircuitAt
  rw [Power.localLength_eq, constraintExponent_eq]

private theorem matrixAssumptionsAt (interface : Interface) (offset : Nat)
    (env : Env) {source : Env}
    (assumptions : Assumptions interface offset source) :
    Power.Assumptions matrixExponent
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env := by
  apply Power.assumptions_of_point_varsBelow
  have offsetLe : offset ≤ matrixOffset interface offset := by
    unfold matrixOffset
    omega
  simpa [matrixPowerInterfaceAt] using
    (interface.gamma offset).varsBelow_mono assumptions.gammaBelow offsetLe

private theorem constraintAssumptionsAt (interface : Interface) (offset : Nat)
    (env : Env) {source : Env}
    (assumptions : Assumptions interface offset source) :
    Power.Assumptions constraintExponent
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env := by
  apply Power.assumptions_of_point_varsBelow
  have offsetLe : offset ≤ constraintOffset interface offset := by
    unfold constraintOffset matrixOffset
    omega
  simpa [constraintPowerInterfaceAt] using
    (interface.gamma offset).varsBelow_mono assumptions.gammaBelow offsetLe

private theorem pointCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env := by
  have callHolds := rows (pointOp interface offset) (by simp [opsAt])
  change (pointCircuitAt interface offset).assumptions offset env →
    (pointCircuitAt interface offset).spec offset env at callHolds
  exact callHolds assumptions.point

private theorem matrixPowerCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Power.SpecHolds matrixExponent (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env := by
  have callHolds := rows (matrixPowerOp interface offset) (by simp [opsAt])
  change (matrixPowerCircuitAt interface offset).assumptions
      (matrixOffset interface offset) env →
    (matrixPowerCircuitAt interface offset).spec
      (matrixOffset interface offset) env at callHolds
  exact callHolds (matrixAssumptionsAt interface offset env assumptions)

private theorem constraintPowerCall_sound (interface : Interface)
    (offset : Nat) (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    Power.SpecHolds constraintExponent
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env := by
  have callHolds := rows (constraintPowerOp interface offset) (by simp [opsAt])
  change (constraintPowerCircuitAt interface offset).assumptions
      (constraintOffset interface offset) env →
    (constraintPowerCircuitAt interface offset).spec
      (constraintOffset interface offset) env at callHolds
  exact callHolds (constraintAssumptionsAt interface offset env assumptions)

private theorem terminalRows_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset)) :
    (interface.terminal offset).eval env =
      (terminalExpr interface offset).eval env := by
  apply (KExpr.equalities_hold_iff env (interface.terminal offset)
    (terminalExpr interface offset)).mp
  intro expression member
  exact rows (Op.assertZero expression) (by
    simp [opsAt, terminalAssertions, member])

private theorem terminalExpr_eval_of_children (interface : Interface)
    (offset : Nat) (env : Env)
    (pointSpec : PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env)
    (matrixSpec : Power.SpecHolds matrixExponent
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env)
    (constraintSpec : Power.SpecHolds constraintExponent
      (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env) :
    (terminalExpr interface offset).eval env =
      referenceTerminal interface offset env := by
  have matrixEq := Power.spec_implies_power matrixExponent
    (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env matrixSpec
  have constraintEq := Power.spec_implies_power constraintExponent
    (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env constraintSpec
  change (gammaMatrixOutput interface offset).eval env =
    TargetPolynomial.power extensionOps.toOps
      ((interface.gamma offset).eval env) matrixExponent at matrixEq
  change (gammaConstraintOutput interface offset).eval env =
    TargetPolynomial.power extensionOps.toOps
      ((interface.gamma offset).eval env) constraintExponent at constraintEq
  unfold gammaMatrixOutput at matrixEq
  unfold gammaConstraintOutput at constraintEq
  unfold PointEquality.Owned.SpecHolds at pointSpec
  unfold terminalExpr referenceTerminal pointEqualityOutput gammaMatrixOutput
    gammaConstraintOutput
  simp only [KExpr.eval_add, KExpr.eval_mul]
  rw [pointSpec, matrixEq, constraintEq]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
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

private theorem evalLeftPoint_eq_of_agree_below (interface : Interface)
    (offset : Nat) (before after : Env)
    (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index) :
    PointEquality.Owned.evalLeftPoint (pointInterfaceAt interface offset)
        offset after =
      PointEquality.Owned.evalLeftPoint (pointInterfaceAt interface offset)
        offset before := by
  apply cubePoint_eq_of_coordinates
  change (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate => (interface.roundPoint offset coordinate).eval after) =
    (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate => (interface.roundPoint offset coordinate).eval before)
  apply List.map_congr_left
  intro coordinate _
  exact (interface.roundPoint offset coordinate).eval_eq_of_agree_below
    offset after before (assumptions.point coordinate).1 agrees

private theorem evalRightPoint_eq_of_agree_below (interface : Interface)
    (offset : Nat) (before after : Env)
    (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index) :
    PointEquality.Owned.evalRightPoint (pointInterfaceAt interface offset)
        offset after =
      PointEquality.Owned.evalRightPoint (pointInterfaceAt interface offset)
        offset before := by
  apply cubePoint_eq_of_coordinates
  change (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate => (interface.alpha offset coordinate).eval after) =
    (canonicalFinIndices productionShape.cubeVariables).map
      (fun coordinate => (interface.alpha offset coordinate).eval before)
  apply List.map_congr_left
  intro coordinate _
  exact (interface.alpha offset coordinate).eval_eq_of_agree_below
    offset after before (assumptions.point coordinate).2 agrees

/-- The semantic final identity is stable when entry wires are unchanged. -/
theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
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

private theorem pointOp_flatConstraints (interface : Interface) (offset : Nat) :
    (pointOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main
        offset) := by
  unfold pointOp pointSubcircuit
  exact FormalCircuit.asSubcircuit_constraints _ _ _

private theorem matrixPowerOp_flatConstraints (interface : Interface)
    (offset : Nat) :
    (matrixPowerOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset)) := by
  unfold matrixPowerOp matrixPowerSubcircuit
  exact FormalCircuit.asSubcircuit_constraints _ _ _

private theorem constraintPowerOp_flatConstraints (interface : Interface)
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

private theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
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

private theorem pointOp_localLength (interface : Interface) (offset : Nat) :
    (pointOp interface offset).localLength = pointLength interface offset := by
  rfl

private theorem matrixPowerOp_localLength (interface : Interface)
    (offset : Nat) :
    (matrixPowerOp interface offset).localLength =
      matrixLength interface offset := by
  rfl

private theorem constraintPowerOp_localLength (interface : Interface)
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

private theorem opsAt_localLength (interface : Interface) (offset : Nat) :
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

private theorem pointRows_preservedAfterMatrix (interface : Interface)
    (offset : Nat) (env afterPoint afterMatrix : Env)
    (assumptions : Assumptions interface offset env)
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
      (pointInterfaceAt interface offset) offset (by
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables])]
    unfold matrixOffset
    rw [pointLength_eq]
    norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · intro index below
    exact matrixAgrees index (Or.inl below)
  · exact pointRows

private theorem pointRows_preservedAfterConstraint (interface : Interface)
    (offset : Nat) (env afterMatrix completed : Env)
    (assumptions : Assumptions interface offset env)
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
      (pointInterfaceAt interface offset) offset (by
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables])]
    unfold constraintOffset matrixOffset
    rw [pointLength_eq, matrixLength_eq]
    norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · intro index below
    exact constraintAgrees index (Or.inl below)
  · exact pointRows

private theorem matrixRows_preservedAfterConstraint (interface : Interface)
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
  · have scope := Power.flatConstraints_varsBelow_exact matrixExponent
      (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) env
      (matrixAssumptionsAt interface offset env assumptions)
    intro expression member
    apply Expr.VarsBelow.mono expression (scope expression member)
    rw [matrixExponent_eq]
    unfold constraintOffset
    rw [matrixLength_eq]
  · intro index below
    exact constraintAgrees index (Or.inl below)
  · exact matrixRows

private theorem terminalRows_complete (interface : Interface) (offset : Nat)
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
  have matrixSpec := Power.soundness matrixExponent
    (matrixPowerInterfaceAt interface offset) completed
      (matrixOffset interface offset)
      (matrixAssumptionsAt interface offset completed assumptions)
      (holdsFlat_implies_holds completed _ matrixRows)
  have constraintSpec := Power.soundness constraintExponent
    (constraintPowerInterfaceAt interface offset) completed
      (constraintOffset interface offset)
      (constraintAssumptionsAt interface offset completed assumptions)
      (holdsFlat_implies_holds completed _ constraintRows)
  apply (KExpr.equalities_hold_iff completed (interface.terminal offset)
    (terminalExpr interface offset)).mpr
  exact semantic.trans
    (terminalExpr_eval_of_children interface offset completed pointSpec
      matrixSpec constraintSpec).symm

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  rcases PointEquality.Owned.build (pointInterfaceAt interface offset) env
      offset assumptions.point with ⟨afterPoint, pointAgrees, pointRows⟩
  rcases Power.build matrixExponent (matrixPowerInterfaceAt interface offset)
      afterPoint (matrixOffset interface offset)
      (matrixAssumptionsAt interface offset afterPoint assumptions) with
    ⟨afterMatrix, matrixAgrees, matrixRows⟩
  have pointMatrixAgrees : AgreesOutside env afterMatrix offset
      (pointLength interface offset + matrixLength interface offset) := by
    have matrixAgreesAt : AgreesOutside afterPoint afterMatrix
        (offset + pointLength interface offset)
        (matrixLength interface offset) := by
      simpa only [matrixOffset] using matrixAgrees
    exact pointAgrees.append matrixAgreesAt
  rcases Power.build constraintExponent
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
    env afterPoint afterMatrix assumptions pointRows matrixAgrees
  have pointRowsCompleted := pointRows_preservedAfterConstraint interface offset
    env afterMatrix completed assumptions pointRowsAfterMatrix constraintAgrees
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

private theorem pointOutput_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (pointEqualityOutput interface offset).VarsBelow
      (matrixOffset interface offset) := by
  have below := PointEquality.Owned.output_varsBelow
    (pointInterfaceAt interface offset) offset env assumptions.point
  unfold pointEqualityOutput
  apply KExpr.varsBelow_mono _ below
  rw [PointEquality.Owned.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset (by
      norm_num [productionShape, Phi81MatrixSource.phi81Shape,
        cubeVariables])]
  unfold matrixOffset
  rw [pointLength_eq]
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem matrixOutput_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (gammaMatrixOutput interface offset).VarsBelow
      (constraintOffset interface offset) := by
  have below := Power.output_varsBelow matrixExponent
    (matrixPowerInterfaceAt interface offset) (matrixOffset interface offset)
      env (matrixAssumptionsAt interface offset env assumptions)
  apply KExpr.varsBelow_mono _ below
  rw [matrixExponent_eq]
  unfold constraintOffset
  rw [matrixLength_eq]

private theorem constraintOutput_varsBelow (interface : Interface)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    (gammaConstraintOutput interface offset).VarsBelow
      (finalOffset interface offset) := by
  have below := Power.output_varsBelow constraintExponent
    (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) env
      (constraintAssumptionsAt interface offset env assumptions)
  apply KExpr.varsBelow_mono _ below
  rw [constraintExponent_eq]
  unfold finalOffset
  rw [constraintLength_eq]

private theorem terminalExpr_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
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
          (matrixOutput_varsBelow interface offset env assumptions)
          (by unfold finalOffset; omega))
        ((interface.eval_A offset).varsBelow_mono assumptions.eval_ABelow
          offsetLe)
    · apply KExpr.mul_varsBelow
      · exact constraintOutput_varsBelow interface offset env assumptions
      · apply KExpr.mul_varsBelow
        · exact KExpr.varsBelow_mono _
            (pointOutput_varsBelow interface offset env assumptions) (by
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

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
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
          (pointInterfaceAt interface offset) offset (by
            norm_num [productionShape, Phi81MatrixSource.phi81Shape,
              cubeVariables])]
        rw [pointLength_eq, matrixLength_eq, constraintLength_eq]
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables]
      · have below := Power.flatConstraints_varsBelow_exact matrixExponent
          (matrixPowerInterfaceAt interface offset)
            (matrixOffset interface offset) env
            (matrixAssumptionsAt interface offset env assumptions)
            expression matrixMember
        apply Expr.VarsBelow.mono expression below
        unfold matrixOffset
        rw [matrixExponent_eq, pointLength_eq, matrixLength_eq,
          constraintLength_eq]
        omega
    · have below := Power.flatConstraints_varsBelow_exact constraintExponent
        (constraintPowerInterfaceAt interface offset)
          (constraintOffset interface offset) env
          (constraintAssumptionsAt interface offset env assumptions)
          expression constraintMember
      apply Expr.VarsBelow.mono expression below
      unfold constraintOffset matrixOffset
      rw [constraintExponent_eq, pointLength_eq, matrixLength_eq,
        constraintLength_eq]
  · exact KExpr.equalities_varsBelow (interface.terminal offset)
      (terminalExpr interface offset)
      (offset + (pointLength interface offset + matrixLength interface offset +
        constraintLength interface offset))
      ((interface.terminal offset).varsBelow_mono assumptions.terminalBelow
        (by omega))
      (by
        have below := terminalExpr_varsBelow interface offset env assumptions
        apply KExpr.varsBelow_mono _ below
        unfold finalOffset constraintOffset matrixOffset
        omega)
      expression terminalMember

/-- Private symbolic variables owned by the fixed production leaf. -/
def privateCount : Nat := 27742

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 27742 := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength, pointLength_eq, matrixLength_eq, constraintLength_eq]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 5 := by
  change (opsAt interface offset).length = 5
  simp [opsAt, terminalAssertions, KExpr.equalities]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      27744 := by
  change (flatConstraints (opsAt interface offset)).length = _
  have pointFlat :
      (flatConstraints (Circuit.ops (pointCircuitAt interface offset).main
        offset)).length = 94 := by
    unfold pointCircuitAt
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      PointEquality.Owned.flatConstraints_length_of_positive
        (pointInterfaceAt interface offset) offset (by
          norm_num [productionShape, Phi81MatrixSource.phi81Shape,
            cubeVariables])
  have matrixFlat :
      (flatConstraints (Circuit.ops (matrixPowerCircuitAt interface offset).main
        (matrixOffset interface offset))).length = 1728 := by
    unfold matrixPowerCircuitAt
    rw [Power.flatConstraints_length, matrixExponent_eq]
  have constraintFlat :
      (flatConstraints (Circuit.ops
        (constraintPowerCircuitAt interface offset).main
        (constraintOffset interface offset))).length = 25920 := by
    unfold constraintPowerCircuitAt
    rw [Power.flatConstraints_length, constraintExponent_eq]
  rw [flatConstraints_opsAt, List.length_append, List.length_append,
    List.length_append, pointFlat, matrixFlat, constraintFlat]
  simp [terminalAssertions, KExpr.equalities]

/-- Concrete parent coverage: the owned child values and terminal assertions
are exactly production `ProtocolPolynomial.terminalFromMessage`. -/
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
  rw [NightstreamFPrime.Spec.Folding.PiCCS.v1_1.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_eq]
  simpa [input, execution, message, matrixExponent, constraintExponent] using
    specification

/-- Exact completeness direction: the canonical SumCheck terminal equality
and the six computed terminal components imply this leaf's specification. -/
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
  rw [NightstreamFPrime.Spec.Folding.PiCCS.v1_1.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_eq]
  simp [extensionOps, matrixExponent, constraintExponent]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity
