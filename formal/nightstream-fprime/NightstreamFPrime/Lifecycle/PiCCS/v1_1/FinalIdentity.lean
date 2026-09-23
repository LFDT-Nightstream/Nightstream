import NightstreamFPrime.Gadgets.Multilinear.PointEquality
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

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
- C2: one opaque shared child for `gamma^864` and `gamma^12960`;
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

def pointCircuitAt (interface : Interface) (parentOffset : Nat) : FormalCircuit :=
  PointEquality.Owned.circuit (pointInterfaceAt interface parentOffset)

def gammaCircuitAt (interface : Interface) (parentOffset : Nat) : FormalCircuit :=
  GammaPowers.circuit (interface.gamma parentOffset)

def pointLength (interface : Interface) (offset : Nat) : Nat :=
  localLength (Circuit.ops (pointCircuitAt interface offset).main offset)

def gammaOffset (interface : Interface) (offset : Nat) : Nat :=
  offset + pointLength interface offset

def gammaLength (interface : Interface) (offset : Nat) : Nat :=
  localLength (Circuit.ops (gammaCircuitAt interface offset).main (gammaOffset interface offset))

def finalOffset (interface : Interface) (offset : Nat) : Nat :=
  gammaOffset interface offset + gammaLength interface offset

def pointEqualityOutput (interface : Interface) (offset : Nat) : KExpr :=
  PointEquality.Owned.output (pointInterfaceAt interface offset) offset

def gammaMatrixOutput (interface : Interface) (offset : Nat) : KExpr :=
  GammaPowers.matrixOutput (interface.gamma offset) (gammaOffset interface offset)

def gammaConstraintOutput (interface : Interface) (offset : Nat) : KExpr :=
  GammaPowers.constraintOutput (interface.gamma offset) (gammaOffset interface offset)

def pointName : String := "piccs.v1_1.final.point_equality"
def gammaName : String := "piccs.v1_1.final.gamma_powers"

def pointSubcircuit (interface : Interface) (offset : Nat) : Subcircuit :=
  (pointCircuitAt interface offset).asSubcircuit pointName offset

def gammaSubcircuit (interface : Interface) (offset : Nat) : Subcircuit :=
  (gammaCircuitAt interface offset).asSubcircuit gammaName (gammaOffset interface offset)

def pointOp (interface : Interface) (offset : Nat) : Op := .subcircuit (pointSubcircuit interface offset)
def gammaOp (interface : Interface) (offset : Nat) : Op := .subcircuit (gammaSubcircuit interface offset)

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
  [pointOp interface offset, gammaOp interface offset] ++
    (terminalAssertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by rfl

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

theorem pointLength_eq (interface : Interface) (offset : Nat) :
    pointLength interface offset = 110 := by
  unfold pointLength pointCircuitAt
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
    PointEquality.Owned.localLength_eq_of_positive
      (pointInterfaceAt interface offset) offset (by
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables])

private theorem gammaLength_eq (interface : Interface) (offset : Nat) :
    gammaLength interface offset = 32 := by
  unfold gammaLength gammaCircuitAt
  exact GammaPowers.localLength_eq _ _

private theorem gammaAssumptionsAt (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    (interface.gamma offset).VarsBelow (gammaOffset interface offset) :=
  KExpr.varsBelow_mono _ assumptions.gammaBelow (Nat.le_add_right _ _)

private theorem pointCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    PointEquality.Owned.SpecHolds
      (pointInterfaceAt interface offset) offset env := by
  have callHolds := rows (pointOp interface offset) (by simp [opsAt])
  change (pointCircuitAt interface offset).assumptions offset env →
    (pointCircuitAt interface offset).spec offset env at callHolds
  exact callHolds assumptions.point

private theorem gammaCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (rows : holds env (opsAt interface offset))
    (assumptions : Assumptions interface offset env) :
    GammaPowers.SpecHolds (interface.gamma offset) (gammaOffset interface offset) env := by
  have callHolds := rows (gammaOp interface offset) (by simp [opsAt])
  change (interface.gamma offset).VarsBelow (gammaOffset interface offset) → _ at callHolds
  exact callHolds (gammaAssumptionsAt interface offset assumptions)

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
    (pointSpec : PointEquality.Owned.SpecHolds (pointInterfaceAt interface offset) offset env)
    (gammaSpec : GammaPowers.SpecHolds (interface.gamma offset) (gammaOffset interface offset) env) :
    (terminalExpr interface offset).eval env = referenceTerminal interface offset env := by
  have matrixEq := gammaSpec.1
  have constraintEq := gammaSpec.2
  unfold PointEquality.Owned.SpecHolds at pointSpec
  unfold terminalExpr referenceTerminal pointEqualityOutput gammaMatrixOutput gammaConstraintOutput
  simp only [KExpr.eval_add, KExpr.eval_mul, matrixExponent_eq, constraintExponent_eq]
  rw [matrixEq, constraintEq, pointSpec]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) : SpecHolds interface offset env := by
  rw [main_ops] at rows
  exact (terminalRows_sound interface offset env rows).trans
    (terminalExpr_eval_of_children interface offset env
      (pointCall_sound interface offset env rows assumptions)
      (gammaCall_sound interface offset env rows assumptions))

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

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest ih =>
    change [expression] ++ flatConstraints (rest.map Op.assertZero) = expression :: rest
    rw [ih]
    rfl

theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      (flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset) ++
        flatConstraints (Circuit.ops (gammaCircuitAt interface offset).main (gammaOffset interface offset))) ++
        terminalAssertions interface offset := by
  have point : (pointOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset) := by
    unfold pointOp pointSubcircuit
    exact FormalCircuit.asSubcircuit_constraints _ _ _
  have gamma : (gammaOp interface offset).flatConstraints =
      flatConstraints (Circuit.ops (gammaCircuitAt interface offset).main
        (gammaOffset interface offset)) := by
    unfold gammaOp gammaSubcircuit
    exact FormalCircuit.asSubcircuit_constraints _ _ _
  have flatten (first second : Op) (checks : List Expr) :
      flatConstraints ([first, second] ++ checks.map Op.assertZero) =
        (first.flatConstraints ++ second.flatConstraints) ++ checks := by
    rw [flatConstraints_append, flatConstraints_assertions]
    change (first.flatConstraints ++ (second.flatConstraints ++ [])) ++ checks = _
    rw [List.append_nil]
  rw [opsAt, flatten, point, gamma]

private theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) = pointLength interface offset + gammaLength interface offset := by
  simp only [opsAt, localLength, List.map_append, List.sum_append, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, List.map_map]
  have assertions : (List.map (Op.localLength ∘ Op.assertZero)
      (terminalAssertions interface offset)).sum = 0 := by
    simp [Function.comp_def, Op.localLength]
  rw [assertions, Nat.add_zero]
  rfl

private theorem pointRows_preservedAfterGamma (interface : Interface)
    (offset : Nat) (env afterPoint afterGamma : Env)
    (assumptions : Assumptions interface offset env)
    (pointRows : holdsFlat afterPoint
      (Circuit.ops (pointCircuitAt interface offset).main offset))
    (gammaAgrees : AgreesOutside afterPoint afterGamma
      (gammaOffset interface offset) (gammaLength interface offset)) :
    holdsFlat afterGamma
      (Circuit.ops (pointCircuitAt interface offset).main offset) := by
  unfold holdsFlat at pointRows ⊢
  apply constraintsHold_of_agree_below afterPoint afterGamma _
    (gammaOffset interface offset)
  · have scope := PointEquality.Owned.flatConstraints_varsBelow
      (pointInterfaceAt interface offset) offset env assumptions.point
    intro expression member
    have below := scope expression member
    apply Expr.VarsBelow.mono expression below
    rw [PointEquality.Owned.program_recipes_length_of_positive
      (pointInterfaceAt interface offset) offset (by
        norm_num [productionShape, Phi81MatrixSource.phi81Shape,
          cubeVariables])]
    unfold gammaOffset
    rw [pointLength_eq]
    norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  · intro index below
    exact gammaAgrees index (Or.inl below)
  · exact pointRows

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed, AgreesOutside env completed offset
      (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  obtain ⟨afterPoint, pointAgrees, pointRows⟩ :=
    PointEquality.Owned.build (pointInterfaceAt interface offset) env offset assumptions.point
  obtain ⟨completed, gammaAgrees, gammaRows⟩ := GammaPowers.build (interface.gamma offset)
    (gammaOffset interface offset) afterPoint (gammaAssumptionsAt interface offset assumptions)
  have gammaAgreesAt : AgreesOutside afterPoint completed
      (offset + pointLength interface offset) (gammaLength interface offset) := by
    simpa only [gammaOffset, gammaLength_eq] using gammaAgrees
  have combinedAgrees : AgreesOutside env completed offset
      (pointLength interface offset + gammaLength interface offset) :=
    pointAgrees.append gammaAgreesAt
  have pointRowsCompleted := pointRows_preservedAfterGamma interface offset env
    afterPoint completed assumptions pointRows gammaAgreesAt
  have pointSpec := PointEquality.Owned.soundness (pointInterfaceAt interface offset)
    completed offset assumptions.point (holdsFlat_implies_holds completed _ pointRowsCompleted)
  have gammaSpec := GammaPowers.soundness (interface.gamma offset) (gammaOffset interface offset)
    completed (holdsFlat_implies_holds completed _ gammaRows)
  have semantic := specHolds_of_agree_below interface offset env completed assumptions
    (fun index below => combinedAgrees index (Or.inl below)) specification
  have terminalRows : ConstraintsHold completed (terminalAssertions interface offset) := by
    apply (KExpr.equalities_hold_iff completed (interface.terminal offset)
      (terminalExpr interface offset)).mpr
    exact semantic.trans (terminalExpr_eval_of_children interface offset completed pointSpec gammaSpec).symm
  refine ⟨completed, ?_, ?_⟩
  · change AgreesOutside env completed offset (localLength (opsAt interface offset))
    rw [opsAt_localLength]
    exact combinedAgrees
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact (constraintsHold_append completed _ _).mpr
      ⟨(constraintsHold_append completed _ _).mpr ⟨pointRowsCompleted, gammaRows⟩, terminalRows⟩

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
      (gammaOffset interface offset) := by
  have below := PointEquality.Owned.output_varsBelow
    (pointInterfaceAt interface offset) offset env assumptions.point
  unfold pointEqualityOutput
  apply KExpr.varsBelow_mono _ below
  rw [PointEquality.Owned.localLength_eq_of_positive
    (pointInterfaceAt interface offset) offset (by
      norm_num [productionShape, Phi81MatrixSource.phi81Shape,
        cubeVariables])]
  unfold gammaOffset
  rw [pointLength_eq]
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem matrixOutput_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (gammaMatrixOutput interface offset).VarsBelow (finalOffset interface offset) := by
  unfold finalOffset
  rw [gammaLength_eq]
  exact GammaPowers.wire_varsBelow _ _ 11 16
    (gammaAssumptionsAt interface offset assumptions) (by decide)

private theorem constraintOutput_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (gammaConstraintOutput interface offset).VarsBelow (finalOffset interface offset) := by
  unfold finalOffset
  rw [gammaLength_eq]
  exact GammaPowers.wire_varsBelow _ _ 16 16
    (gammaAssumptionsAt interface offset assumptions) (by decide)

private theorem terminalExpr_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    (terminalExpr interface offset).VarsBelow (finalOffset interface offset) := by
  have offsetLe : offset ≤ finalOffset interface offset := by
    unfold finalOffset gammaOffset
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
              unfold finalOffset gammaOffset
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
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + localLength (opsAt interface offset))
  rw [flatConstraints_opsAt, opsAt_localLength]
  intro expression member
  rcases List.mem_append.mp member with coreMember | terminalMember
  · rcases List.mem_append.mp coreMember with pointMember | gammaMember
    · have below := PointEquality.Owned.flatConstraints_varsBelow
        (pointInterfaceAt interface offset) offset env assumptions.point expression pointMember
      apply Expr.VarsBelow.mono expression below
      rw [PointEquality.Owned.program_recipes_length_of_positive
        (pointInterfaceAt interface offset) offset (by
          norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables])]
      rw [pointLength_eq, gammaLength_eq]
      norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
    · have below := GammaPowers.flatConstraints_varsBelow (interface.gamma offset)
        (gammaOffset interface offset) (gammaAssumptionsAt interface offset assumptions)
        expression gammaMember
      simpa only [gammaOffset, gammaLength_eq, Nat.add_assoc] using below
  · apply KExpr.equalities_varsBelow (interface.terminal offset) (terminalExpr interface offset)
      (offset + (pointLength interface offset + gammaLength interface offset))
      ((interface.terminal offset).varsBelow_mono assumptions.terminalBelow (by omega))
      _ expression terminalMember
    simpa only [finalOffset, gammaOffset, Nat.add_assoc] using
      terminalExpr_varsBelow interface offset env assumptions

def privateCount : Nat := 142

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 142 := by
  change localLength (opsAt interface offset) = _
  rw [opsAt_localLength, pointLength_eq, gammaLength_eq]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 4 := by
  simp [circuit, main, Circuit.ops, opsAt, terminalAssertions, KExpr.equalities]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length = 144 := by
  change (flatConstraints (opsAt interface offset)).length = _
  have pointFlat : (flatConstraints (Circuit.ops (pointCircuitAt interface offset).main offset)).length = 110 := by
    unfold pointCircuitAt
    simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
      PointEquality.Owned.flatConstraints_length_of_positive
        (pointInterfaceAt interface offset) offset (by
          norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables])
  rw [flatConstraints_opsAt, List.length_append, List.length_append, pointFlat]
  change 110 + (flatConstraints (Circuit.ops (GammaPowers.circuit (interface.gamma offset)).main
    (gammaOffset interface offset))).length + _ = _
  rw [GammaPowers.flatConstraints_length]
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
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
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
  rw [NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.terminal_eq_eval_K_add_shifted_eval_A_add_constraints]
  unfold SignedJointIdentity.gammaTerm
  rw [gammaFreshPower_eq]
  simp [extensionOps, matrixExponent, constraintExponent]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity
