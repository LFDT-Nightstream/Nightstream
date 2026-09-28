import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
Owns the Stage 1 running-instance branch.

The base branch selects the canonical HyperNova `defaultRunning` value. The
recursive branch selects the complete PiDEC running output. One
non-authoritative inverse-or-zero hint derives the branch flag. An explicit
binding row and one mux row per canonical running word make the hint
non-authoritative.

This leaf does not own PiDEC validity, application execution, state hashing,
physical column placement, or terminal acceptance.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.RunningTransition

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.GoldilocksPrime
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1

def exactWordCount : Nat := 49353
def stateWordCount : Nat := 4
def exactPrivateCount : Nat := 1
def exactRowCount : Nat := 49358

abbrev WordIndex := Fin exactWordCount
abbrev StateIndex := Fin stateWordCount

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  iteration : Nat → Expr
  initialState : Nat → StateIndex → Expr
  currentState : Nat → StateIndex → Expr
  recursive : Nat → StatementAbsorption.RunningExpr logicalWidth publicFits
  output : Nat → StatementAbsorption.RunningExpr logicalWidth publicFits

def iterationValue {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : F :=
  (interface.iteration offset).eval env

def runningWord {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (index : WordIndex) : Expr :=
  (StatementAbsorption.serializeRunningExpr running).getD index.val 0

def defaultWord {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (index : WordIndex) : F :=
  (serializeRunning (logicalWidth := logicalWidth) (publicFits := publicFits)
    (defaultRunning (logicalWidth := logicalWidth) (publicFits := publicFits))).getD
      index.val 0

def inverseExpr (offset : Nat) : Expr :=
  Expr.var offset

def inverseHint {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Hint :=
  .inverseOrZero (interface.iteration offset)

def recursiveFlag {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Expr :=
  interface.iteration offset * inverseExpr offset

def baseFlag {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Expr :=
  1 - recursiveFlag interface offset

/-- This row forces the recursive flag to one when iteration is nonzero. -/
def bindingConstraint {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Expr :=
  interface.iteration offset * baseFlag interface offset

def muxConstraint {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (index : WordIndex) : Expr :=
  (baseFlag interface offset * Expr.const
        (defaultWord (logicalWidth := logicalWidth)
          (publicFits := publicFits) index) +
      recursiveFlag interface offset *
        runningWord (interface.recursive offset) index) -
    runningWord (interface.output offset) index

def muxConstraints {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  List.ofFn (muxConstraint interface offset)

/-- HyperNova Construction 2's base branch requires `z0 = zi`. The existing
base flag gates the four fixed application-state words without another branch
selector. -/
def baseStateConstraint {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (index : StateIndex) : Expr :=
  baseFlag interface offset *
    (interface.initialState offset index - interface.currentState offset index)

def baseStateConstraints {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Expr :=
  List.ofFn (baseStateConstraint interface offset)

@[simp] theorem muxConstraints_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (muxConstraints interface offset).length = exactWordCount := by
  simp [muxConstraints]

@[simp] theorem baseStateConstraints_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (baseStateConstraints interface offset).length = stateWordCount := by
  simp [baseStateConstraints]

private theorem zipWith3_eq_ofFn_getD
    {Alpha Beta Gamma Delta : Type}
    (combine : Alpha → Beta → Gamma → Delta)
    (fallbackAlpha : Alpha) (fallbackBeta : Beta) (fallbackGamma : Gamma) :
    ∀ (count : Nat) (alpha : List Alpha) (beta : List Beta)
      (gamma : List Gamma),
      alpha.length = count → beta.length = count →
        gamma.length = count →
        List.zipWith3 combine alpha beta gamma =
          List.ofFn fun index : Fin count =>
            combine (alpha.getD index.val fallbackAlpha)
              (beta.getD index.val fallbackBeta)
              (gamma.getD index.val fallbackGamma) := by
  intro count
  induction count with
  | zero =>
      intro alpha beta gamma alphaLength betaLength gammaLength
      have alphaEmpty : alpha = [] := List.length_eq_zero_iff.mp alphaLength
      have betaEmpty : beta = [] := List.length_eq_zero_iff.mp betaLength
      have gammaEmpty : gamma = [] := List.length_eq_zero_iff.mp gammaLength
      subst alpha
      subst beta
      subst gamma
      rfl
  | succ count inductionHypothesis =>
      intro alpha beta gamma alphaLength betaLength gammaLength
      cases alpha with
      | nil => simp at alphaLength
      | cons alphaHead alphaTail =>
          cases beta with
          | nil => simp at betaLength
          | cons betaHead betaTail =>
              cases gamma with
              | nil => simp at gammaLength
              | cons gammaHead gammaTail =>
                  have alphaTailLength : alphaTail.length = count := by
                    simpa using alphaLength
                  have betaTailLength : betaTail.length = count := by
                    simpa using betaLength
                  have gammaTailLength : gammaTail.length = count := by
                    simpa using gammaLength
                  rw [List.ofFn_succ]
                  simp only [List.zipWith3]
                  rw [inductionHypothesis alphaTail betaTail gammaTail
                    alphaTailLength betaTailLength gammaTailLength]
                  simp

def muxConstraintsFast {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  List.zipWith3
    (fun default recursive output =>
      (baseFlag interface offset * Expr.const default +
        recursiveFlag interface offset * recursive) - output)
    (serializeRunning (logicalWidth := logicalWidth) (publicFits := publicFits)
      (defaultRunning (logicalWidth := logicalWidth)
        (publicFits := publicFits)))
    (StatementAbsorption.serializeRunningExpr (interface.recursive offset))
    (StatementAbsorption.serializeRunningExpr (interface.output offset))

theorem muxConstraintsFast_eq_muxConstraints {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    muxConstraintsFast interface offset = muxConstraints interface offset := by
  rw [muxConstraintsFast, muxConstraints]
  rw [zipWith3_eq_ofFn_getD _ (0 : F) (0 : Expr) (0 : Expr)
    exactWordCount]
  · rfl
  · simp [exactWordCount, serializeRunning_length]
  · simp [exactWordCount,
      StatementAbsorption.serializeRunningExpr_length]
  · simp [exactWordCount,
      StatementAbsorption.serializeRunningExpr_length]

def constraints {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  bindingConstraint interface offset ::
    (muxConstraints interface offset ++ baseStateConstraints interface offset)

def constraintsFast {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    List Expr :=
  bindingConstraint interface offset ::
    (muxConstraintsFast interface offset ++ baseStateConstraints interface offset)

theorem constraintsFast_eq_constraints {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    constraintsFast interface offset = constraints interface offset := by
  rw [constraintsFast, constraints, muxConstraintsFast_eq_muxConstraints]

def operations {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Op :=
  .witness (WitnessBatch.hinted offset [inverseHint interface offset]) ::
    (constraints interface offset).map .assertZero

def main {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : Circuit Unit :=
  fun offset => ((), offset + exactPrivateCount, operations interface offset)

structure Assumptions {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (_env : Env) : Prop where
  iteration : (interface.iteration offset).VarsBelow offset
  initialState : ∀ index,
    (interface.initialState offset index).VarsBelow offset
  currentState : ∀ index,
    (interface.currentState offset index).VarsBelow offset
  recursive : ∀ index,
    (runningWord (interface.recursive offset) index).VarsBelow offset
  output : ∀ index,
    (runningWord (interface.output offset) index).VarsBelow offset

/-- Field-level scope certificate for one complete symbolic running vector. -/
structure RunningBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (bound : Nat) : Prop where
  point : ∀ coordinate, (running.point coordinate).VarsBelow bound
  commitment : ∀ source row coefficient,
    (running.commitment source row coefficient).VarsBelow bound
  publicInput : ∀ source column,
    (running.publicInput source column).VarsBelow bound
  eval_K : ∀ source coefficient,
    (running.evaluation source).eval_K coefficient |>.VarsBelow bound
  eval_A : ∀ source matrix coefficient,
    (running.evaluation source).eval_A matrix coefficient |>.VarsBelow bound

theorem RunningBelow.mono {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {running : StatementAbsorption.RunningExpr logicalWidth publicFits}
    {bound larger : Nat} (below : RunningBelow running bound)
    (le : bound ≤ larger) : RunningBelow running larger where
  point coordinate :=
    ⟨Expr.VarsBelow.mono _ (below.point coordinate).1 le,
      Expr.VarsBelow.mono _ (below.point coordinate).2 le⟩
  commitment source row coefficient :=
    (below.commitment source row coefficient).mono _ le
  publicInput source column := (below.publicInput source column).mono _ le
  eval_K source coefficient :=
    ⟨Expr.VarsBelow.mono _ (below.eval_K source coefficient).1 le,
      Expr.VarsBelow.mono _ (below.eval_K source coefficient).2 le⟩
  eval_A source matrix coefficient :=
    ⟨Expr.VarsBelow.mono _ (below.eval_A source matrix coefficient).1 le,
      Expr.VarsBelow.mono _ (below.eval_A source matrix coefficient).2 le⟩

private theorem serializeKExpr_varsBelow (value : KExpr) (bound : Nat)
    (below : value.VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.serializeKExpr value,
      expression.VarsBelow bound := by
  intro expression member
  simp only [StatementAbsorption.serializeKExpr, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact below.1
  · exact below.2

private theorem serializePointExpr_varsBelow
    (point : Fin productionShape.cubeVariables → KExpr) (bound : Nat)
    (below : ∀ coordinate, (point coordinate).VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.serializePointExpr point,
      expression.VarsBelow bound := by
  intro expression member
  rw [StatementAbsorption.serializePointExpr, List.mem_flatMap] at member
  rcases member with ⟨coordinate, _coordinateMember, expressionMember⟩
  exact serializeKExpr_varsBelow (point coordinate) bound (below coordinate)
    expression expressionMember

private theorem serializeCommitmentExpr_varsBelow
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) (bound : Nat)
    (below : ∀ row coefficient,
      (commitment row coefficient).VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.serializeCommitmentExpr commitment,
      expression.VarsBelow bound := by
  intro expression member
  rw [StatementAbsorption.serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _rowMember, expressionMember⟩
  rw [List.mem_map] at expressionMember
  rcases expressionMember with ⟨coefficient, _coefficientMember, rfl⟩
  exact below row coefficient

private theorem serializePublicInputExpr_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (bound : Nat) (below : ∀ column, (input column).VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.serializePublicInputExpr input,
      expression.VarsBelow bound := by
  intro expression member
  rw [StatementAbsorption.serializePublicInputExpr, List.mem_map] at member
  rcases member with ⟨column, _columnMember, rfl⟩
  exact below column

private theorem serializeEvaluationExpr_varsBelow
    (evaluation : StatementAbsorption.EvaluationExpr) (bound : Nat)
    (eval_K : ∀ coefficient,
      (evaluation.eval_K coefficient).VarsBelow bound)
    (eval_A : ∀ matrix coefficient,
      (evaluation.eval_A matrix coefficient).VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.serializeEvaluationExpr evaluation,
      expression.VarsBelow bound := by
  intro expression member
  rw [StatementAbsorption.serializeEvaluationExpr, List.mem_append] at member
  rcases member with padMember | matrixMember
  · rw [List.mem_flatMap] at padMember
    rcases padMember with ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_varsBelow (evaluation.eval_K coefficient) bound
      (eval_K coefficient)
      expression expressionMember
  · rw [List.mem_flatMap] at matrixMember
    rcases matrixMember with ⟨matrix, _matrixMember, coefficientMember⟩
    rw [List.mem_flatMap] at coefficientMember
    rcases coefficientMember with
      ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_varsBelow
      (evaluation.eval_A matrix coefficient) bound (eval_A matrix coefficient)
      expression expressionMember

private theorem blockExpr_varsBelow (words : List Expr) (bound : Nat)
    (below : ∀ expression ∈ words, expression.VarsBelow bound) :
    ∀ expression ∈ StatementAbsorption.blockExpr words,
      expression.VarsBelow bound := by
  intro expression member
  simp only [StatementAbsorption.blockExpr, List.mem_cons] at member
  rcases member with rfl | wordMember
  · trivial
  · exact below expression wordMember

theorem serializeRunningExpr_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (bound : Nat) (below : RunningBelow running bound) :
    ∀ expression ∈ StatementAbsorption.serializeRunningExpr running,
      expression.VarsBelow bound := by
  intro expression member
  rw [StatementAbsorption.serializeRunningExpr, List.mem_append] at member
  rcases member with pointMember | groupMember
  · exact blockExpr_varsBelow _ bound
      (serializePointExpr_varsBelow running.point bound below.point)
      expression pointMember
  · rw [List.mem_flatMap] at groupMember
    rcases groupMember with ⟨source, _sourceMember, expressionMember⟩
    simp only [List.mem_append] at expressionMember
    rcases expressionMember with (commitmentMember | publicMember) |
      evaluationMember
    · exact blockExpr_varsBelow _ bound
        (serializeCommitmentExpr_varsBelow (running.commitment source) bound
          (below.commitment source)) expression commitmentMember
    · exact blockExpr_varsBelow _ bound
        (serializePublicInputExpr_varsBelow (running.publicInput source) bound
          (below.publicInput source)) expression publicMember
    · exact blockExpr_varsBelow _ bound
        (serializeEvaluationExpr_varsBelow (running.evaluation source) bound
          (below.eval_K source) (below.eval_A source))
        expression evaluationMember

theorem runningWord_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (bound : Nat) (below : RunningBelow running bound) (index : WordIndex) :
    (runningWord running index).VarsBelow bound := by
  have indexBound : index.val <
      (StatementAbsorption.serializeRunningExpr running).length := by
    rw [StatementAbsorption.serializeRunningExpr_length]
    exact index.isLt
  rw [runningWord, List.getD_eq_get _ _ ⟨index.val, indexBound⟩]
  exact serializeRunningExpr_varsBelow running bound below _
    (List.get_mem _ ⟨index.val, indexBound⟩)

def WordsEqual {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) : Prop :=
  ∀ index, (runningWord left index).eval env =
    (runningWord right index).eval env

def WordsEqualDefault {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) : Prop :=
  ∀ index, (runningWord running index).eval env =
    defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits) index

structure SpecHolds {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop where
  initialState : iterationValue interface offset env = 0 → ∀ index,
    (interface.initialState offset index).eval env =
      (interface.currentState offset index).eval env
  base : iterationValue interface offset env = 0 →
    WordsEqualDefault (interface.output offset) env
  recursive : iterationValue interface offset env ≠ 0 →
    WordsEqual (interface.output offset) (interface.recursive offset) env

/-- Running-transition semantics depend only on the declared expressions
below the child start. -/
theorem specHolds_of_agree_below
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (before after : Env)
    (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have iterationEq : iterationValue interface offset after =
      iterationValue interface offset before :=
    Expr.eval_eq_of_agree_below _ offset after before assumptions.iteration
      agrees
  refine {
    initialState := ?_
    base := ?_
    recursive := ?_ }
  · intro afterZero index
    have beforeZero : iterationValue interface offset before = 0 := by
      rw [← iterationEq]
      exact afterZero
    calc
      (interface.initialState offset index).eval after =
          (interface.initialState offset index).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.initialState index) agrees
      _ = (interface.currentState offset index).eval before :=
        specification.initialState beforeZero index
      _ = (interface.currentState offset index).eval after :=
        (Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.currentState index) agrees).symm
  · intro afterZero index
    have beforeZero : iterationValue interface offset before = 0 := by
      rw [← iterationEq]
      exact afterZero
    calc
      (runningWord (interface.output offset) index).eval after =
          (runningWord (interface.output offset) index).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.output index) agrees
      _ = defaultWord (logicalWidth := logicalWidth)
          (publicFits := publicFits) index := specification.base beforeZero index
  · intro afterNonzero index
    have beforeNonzero : iterationValue interface offset before ≠ 0 := by
      intro beforeZero
      apply afterNonzero
      rw [iterationEq, beforeZero]
    calc
      (runningWord (interface.output offset) index).eval after =
          (runningWord (interface.output offset) index).eval before :=
        Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.output index) agrees
      _ = (runningWord (interface.recursive offset) index).eval before :=
        specification.recursive beforeNonzero index
      _ = (runningWord (interface.recursive offset) index).eval after :=
        (Expr.eval_eq_of_agree_below _ offset after before
          (assumptions.recursive index) agrees).symm

/-- Running-transition semantics transport through equality of every value
that the branch predicate reads. This is weaker and more exact than requiring
the complete prefix below the child start to remain unchanged. -/
theorem specHolds_of_values_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (before after : Env)
    (iterationEq : iterationValue interface offset before =
      iterationValue interface offset after)
    (initialStateEq : ∀ index,
      (interface.initialState offset index).eval before =
        (interface.initialState offset index).eval after)
    (currentStateEq : ∀ index,
      (interface.currentState offset index).eval before =
        (interface.currentState offset index).eval after)
    (recursiveEq : ∀ index,
      (runningWord (interface.recursive offset) index).eval before =
        (runningWord (interface.recursive offset) index).eval after)
    (outputEq : ∀ index,
      (runningWord (interface.output offset) index).eval before =
        (runningWord (interface.output offset) index).eval after)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  refine {
    initialState := ?_
    base := ?_
    recursive := ?_ }
  · intro afterZero index
    have beforeZero : iterationValue interface offset before = 0 := by
      rw [iterationEq]
      exact afterZero
    calc
      (interface.initialState offset index).eval after =
          (interface.initialState offset index).eval before :=
        (initialStateEq index).symm
      _ = (interface.currentState offset index).eval before :=
        specification.initialState beforeZero index
      _ = (interface.currentState offset index).eval after :=
        currentStateEq index
  · intro afterZero index
    have beforeZero : iterationValue interface offset before = 0 := by
      rw [iterationEq]
      exact afterZero
    calc
      (runningWord (interface.output offset) index).eval after =
          (runningWord (interface.output offset) index).eval before :=
        (outputEq index).symm
      _ = defaultWord (logicalWidth := logicalWidth)
          (publicFits := publicFits) index :=
        specification.base beforeZero index
  · intro afterNonzero index
    have beforeNonzero : iterationValue interface offset before ≠ 0 := by
      intro beforeZero
      apply afterNonzero
      rw [← iterationEq]
      exact beforeZero
    calc
      (runningWord (interface.output offset) index).eval after =
          (runningWord (interface.output offset) index).eval before :=
        (outputEq index).symm
      _ = (runningWord (interface.recursive offset) index).eval before :=
        specification.recursive beforeNonzero index
      _ = (runningWord (interface.recursive offset) index).eval after :=
        recursiveEq index

private theorem flatConstraints_assertions (items : List Expr) :
    flatConstraints (items.map .assertZero) = items := by
  induction items with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change expression :: flatConstraints (rest.map .assertZero) =
        expression :: rest
      rw [inductionHypothesis]

private theorem localLength_assertions (items : List Expr) :
    localLength (items.map .assertZero) = 0 := by
  induction items with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map .assertZero) = 0
      simpa using inductionHypothesis

theorem flatConstraints_operations {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    flatConstraints (operations interface offset) =
      constraints interface offset := by
  change recipeConstraints offset [] ++
      flatConstraints ((constraints interface offset).map .assertZero) =
    constraints interface offset
  rw [flatConstraints_assertions]
  rfl

theorem localLength_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (operations interface offset) = exactPrivateCount := by
  change 1 + localLength
      ((constraints interface offset).map .assertZero) = exactPrivateCount
  rw [localLength_assertions]
  rfl

theorem flatConstraints_length_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = exactRowCount := by
  rw [flatConstraints_operations]
  rw [constraints, List.length_cons, List.length_append,
    muxConstraints_length, baseStateConstraints_length]
  rfl

private theorem inverseExpr_varsBelow (offset : Nat) :
    (inverseExpr offset).VarsBelow (offset + exactPrivateCount) := by
  simp [inverseExpr, Expr.VarsBelow, exactPrivateCount]

private theorem recursiveFlag_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    (recursiveFlag interface offset).VarsBelow
      (offset + exactPrivateCount) := by
  exact Expr.VarsBelow.mul _ _ _
    (Expr.VarsBelow.mono (interface.iteration offset)
      (lower := offset) (upper := offset + exactPrivateCount)
      assumptions.iteration (by omega))
    (inverseExpr_varsBelow offset)

private theorem baseFlag_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    (baseFlag interface offset).VarsBelow
      (offset + exactPrivateCount) := by
  exact Expr.VarsBelow.sub _ _ _ trivial
    (recursiveFlag_varsBelow interface offset assumptions)

theorem flatConstraints_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + exactPrivateCount) := by
  rw [flatConstraints_operations]
  intro expression member
  rcases List.mem_cons.mp member with bindingMember | muxMember
  · subst expression
    exact Expr.VarsBelow.mul _ _ _
      (Expr.VarsBelow.mono (interface.iteration offset)
        (lower := offset) (upper := offset + exactPrivateCount)
        assumptions.iteration (by omega))
      (baseFlag_varsBelow interface offset assumptions)
  · rcases List.mem_append.mp muxMember with muxMember | stateMember
    · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
      have outputBelow := Expr.VarsBelow.mono
        (runningWord (interface.output offset) index)
        (lower := offset) (upper := offset + exactPrivateCount)
        (assumptions.output index) (by omega)
      have recursiveBelow := Expr.VarsBelow.mono
        (runningWord (interface.recursive offset) index)
        (lower := offset) (upper := offset + exactPrivateCount)
        (assumptions.recursive index) (by omega)
      exact Expr.VarsBelow.sub _ _ _
        (Expr.VarsBelow.add _ _ _
          (Expr.VarsBelow.mul _ _ _
            (baseFlag_varsBelow interface offset assumptions) trivial)
          (Expr.VarsBelow.mul _ _ _
            (recursiveFlag_varsBelow interface offset assumptions)
            recursiveBelow)) outputBelow
    · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
      have initialBelow := Expr.VarsBelow.mono
        (interface.initialState offset index)
        (lower := offset) (upper := offset + exactPrivateCount)
        (assumptions.initialState index) (by omega)
      have currentBelow := Expr.VarsBelow.mono
        (interface.currentState offset index)
        (lower := offset) (upper := offset + exactPrivateCount)
        (assumptions.currentState index) (by omega)
      exact Expr.VarsBelow.mul _ _ _
        (baseFlag_varsBelow interface offset assumptions)
        (Expr.VarsBelow.sub _ _ _ initialBelow currentBelow)

private theorem constraintsHold_of_holds {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (rows : holds env (operations interface offset)) :
    ConstraintsHold env (constraints interface offset) := by
  intro expression member
  exact rows (.assertZero expression) (by simp [operations, member])

@[simp] private theorem exprOne_eval (env : Env) :
    ((1 : Expr).eval env) = (1 : F) := by
  rfl

private theorem recursiveFlag_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) :
    (recursiveFlag interface offset).eval env =
      iterationValue interface offset env * env offset := by
  rfl

private theorem baseFlag_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) :
    (baseFlag interface offset).eval env =
      1 - (recursiveFlag interface offset).eval env := by
  simp [baseFlag]

private theorem bindingConstraint_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) :
    (bindingConstraint interface offset).eval env =
      iterationValue interface offset env *
        (baseFlag interface offset).eval env := by
  rfl

private theorem muxConstraint_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (index : WordIndex) (env : Env) :
    (muxConstraint interface offset index).eval env =
      ((baseFlag interface offset).eval env *
            defaultWord (logicalWidth := logicalWidth)
              (publicFits := publicFits) index +
          (recursiveFlag interface offset).eval env *
            (runningWord (interface.recursive offset) index).eval env) -
        (runningWord (interface.output offset) index).eval env := by
  simp [muxConstraint]

private theorem baseStateConstraint_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (index : StateIndex) (env : Env) :
    (baseStateConstraint interface offset index).eval env =
      (baseFlag interface offset).eval env *
        ((interface.initialState offset index).eval env -
          (interface.currentState offset index).eval env) := by
  simp [baseStateConstraint]

theorem soundness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have constraintRows := constraintsHold_of_holds interface offset env rows
  refine {
    initialState := ?_
    base := ?_
    recursive := ?_ }
  · intro iterationZero index
    have row := constraintRows (baseStateConstraint interface offset index) (by
      rw [constraints]
      exact List.mem_cons_of_mem _
        (List.mem_append_right _ (List.mem_ofFn.mpr ⟨index, rfl⟩)))
    have recursiveZero :
        (recursiveFlag interface offset).eval env = 0 := by
      rw [recursiveFlag_eval, iterationZero, zero_mul]
    have baseOne : (baseFlag interface offset).eval env = 1 := by
      rw [baseFlag_eval, recursiveZero, sub_zero]
    apply sub_eq_zero.mp
    simpa [baseStateConstraint_eval, baseOne] using row
  · intro iterationZero index
    have row := constraintRows (muxConstraint interface offset index) (by
      rw [constraints]
      exact List.mem_cons_of_mem _
        (List.mem_append_left _ (List.mem_ofFn.mpr ⟨index, rfl⟩)))
    have recursiveZero :
        (recursiveFlag interface offset).eval env = 0 := by
      rw [recursiveFlag_eval, iterationZero, zero_mul]
    have baseOne : (baseFlag interface offset).eval env = 1 := by
      rw [baseFlag_eval, recursiveZero, sub_zero]
    symm
    apply sub_eq_zero.mp
    simpa [muxConstraint_eval, baseOne, recursiveZero]
      using row
  · intro iterationNonzero index
    have bindingRow := constraintRows (bindingConstraint interface offset) (by
      simp [constraints])
    have product : iterationValue interface offset env *
        (baseFlag interface offset).eval env = 0 := by
      simpa [bindingConstraint_eval] using bindingRow
    have baseZero : (baseFlag interface offset).eval env = 0 := by
      rcases baseFieldNoZeroDivisors _ _ product with impossible | zero
      · exact False.elim (iterationNonzero impossible)
      · exact zero
    have recursiveOne :
        (recursiveFlag interface offset).eval env = 1 := by
      have equal : (1 : F) = (recursiveFlag interface offset).eval env :=
        sub_eq_zero.mp (by simpa [baseFlag_eval] using baseZero)
      exact equal.symm
    have row := constraintRows (muxConstraint interface offset index) (by
      rw [constraints]
      exact List.mem_cons_of_mem _
        (List.mem_append_left _ (List.mem_ofFn.mpr ⟨index, rfl⟩)))
    symm
    apply sub_eq_zero.mp
    simpa [muxConstraint_eval, baseZero, recursiveOne]
      using row

def completeEnv {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) : Env :=
  Env.set env offset (Hint.inverse (iterationValue interface offset env))

private theorem completed_agrees_below {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset index : Nat) (below : index < offset) :
    completeEnv interface env offset index = env index := by
  exact Env.set_of_ne env offset index _ (by omega)

private theorem completed_iteration {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env) :
    iterationValue interface offset (completeEnv interface env offset) =
      iterationValue interface offset env := by
  exact Expr.eval_eq_of_agree_below _ offset _ _ assumptions.iteration
    (completed_agrees_below interface env offset)

private theorem completed_runningWord {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (index : WordIndex) :
    (runningWord (interface.recursive offset) index).eval
        (completeEnv interface env offset) =
      (runningWord (interface.recursive offset) index).eval env := by
  exact Expr.eval_eq_of_agree_below _ offset _ _
    (assumptions.recursive index)
    (completed_agrees_below interface env offset)

private theorem completed_initialState {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (index : StateIndex) :
    (interface.initialState offset index).eval
        (completeEnv interface env offset) =
      (interface.initialState offset index).eval env := by
  exact Expr.eval_eq_of_agree_below _ offset _ _
    (assumptions.initialState index)
    (completed_agrees_below interface env offset)

private theorem completed_currentState {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (index : StateIndex) :
    (interface.currentState offset index).eval
        (completeEnv interface env offset) =
      (interface.currentState offset index).eval env := by
  exact Expr.eval_eq_of_agree_below _ offset _ _
    (assumptions.currentState index)
    (completed_agrees_below interface env offset)

private theorem completed_outputWord {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (index : WordIndex) :
    (runningWord (interface.output offset) index).eval
        (completeEnv interface env offset) =
      (runningWord (interface.output offset) index).eval env := by
  exact Expr.eval_eq_of_agree_below _ offset _ _
    (assumptions.output index)
    (completed_agrees_below interface env offset)

private theorem completed_inverse {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) :
    (inverseExpr offset).eval (completeEnv interface env offset) =
      Hint.inverse (iterationValue interface offset env) := by
  simp [inverseExpr, completeEnv]

private theorem mul_hintInverse_eq_one (value : F) (nonzero : value ≠ 0) :
    value * Hint.inverse value = 1 := by
  have valuePositive : 0 < value.val := Nat.pos_of_ne_zero (by
    intro valueValZero
    apply nonzero
    apply Fin.eq_of_val_eq
    simpa using valueValZero)
  have notDvd : ¬goldilocksModulus ∣ value.val := by
    intro divides
    have lower := Nat.le_of_dvd valuePositive divides
    exact (not_le_of_gt value.isLt) lower
  have coprime : Nat.Coprime value.val goldilocksModulus :=
    (goldilocks_natPrime.coprime_iff_not_dvd.mpr notDvd).symm
  unfold Hint.inverse
  have gcdOne : Nat.gcd value.val goldilocksModulus = 1 :=
    Nat.coprime_iff_gcd_eq_one.mp coprime
  unfold F goldilocksModulus at value gcdOne ⊢
  have law := ZMod.mul_inv_eq_gcd
    (n := 18446744069414584321) value
  dsimp [ZMod, ZMod.val] at law
  rw [gcdOne] at law
  exact law

private theorem completed_recursiveFlag {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env) :
    (recursiveFlag interface offset).eval (completeEnv interface env offset) =
      iterationValue interface offset env *
        Hint.inverse (iterationValue interface offset env) := by
  rw [recursiveFlag_eval, completed_iteration interface env offset assumptions]
  simp [completeEnv]

private theorem completeEnv_holdsFlat {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    holdsFlat (completeEnv interface env offset)
      (operations interface offset) := by
  unfold holdsFlat
  rw [flatConstraints_operations]
  intro expression member
  rcases List.mem_cons.mp member with bindingMember | muxMember
  · subst expression
    rw [bindingConstraint_eval,
      completed_iteration interface env offset assumptions]
    by_cases iterationZero : iterationValue interface offset env = 0
    · rw [iterationZero, zero_mul]
    · rw [baseFlag_eval, completed_recursiveFlag interface env offset assumptions,
        mul_hintInverse_eq_one _ iterationZero, sub_self, mul_zero]
  · rcases List.mem_append.mp muxMember with muxMember | stateMember
    · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
      rw [muxConstraint_eval,
        completed_outputWord interface env offset assumptions index,
        completed_runningWord interface env offset assumptions index]
      by_cases iterationZero : iterationValue interface offset env = 0
      · have selected := specification.base iterationZero index
        rw [baseFlag_eval,
          completed_recursiveFlag interface env offset assumptions,
          iterationZero, zero_mul, sub_zero, one_mul, zero_mul, add_zero,
          selected, sub_self]
      · have selected := specification.recursive iterationZero index
        rw [baseFlag_eval,
          completed_recursiveFlag interface env offset assumptions,
          mul_hintInverse_eq_one _ iterationZero, sub_self, zero_mul, one_mul,
          zero_add, selected, sub_self]
    · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
      rw [baseStateConstraint_eval,
        completed_initialState interface env offset assumptions index,
        completed_currentState interface env offset assumptions index]
      by_cases iterationZero : iterationValue interface offset env = 0
      · have selected := specification.initialState iterationZero index
        rw [baseFlag_eval,
          completed_recursiveFlag interface env offset assumptions,
          iterationZero, zero_mul, sub_zero, selected, sub_self, mul_zero]
      · rw [baseFlag_eval,
          completed_recursiveFlag interface env offset assumptions,
          mul_hintInverse_eq_one _ iterationZero, sub_self, zero_mul]

theorem completeness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  refine ⟨completeEnv interface env offset, ?_,
    completeEnv_holdsFlat interface env offset assumptions specification⟩
  rw [localLength_eq]
  intro index outside
  unfold completeEnv
  apply Env.set_of_ne
  simp [exactPrivateCount] at outside
  omega

theorem runningWord_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) (index : WordIndex) :
    (runningWord running index).eval env =
      (serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning running env)).getD index.val 0 := by
  rw [← StatementAbsorption.serializeRunningExpr_eval running env]
  change ((StatementAbsorption.serializeRunningExpr running).getD
      index.val 0).eval env =
    ((StatementAbsorption.serializeRunningExpr running).map
      (Expr.eval env)).getD index.val 0
  exact (List.getD_map
    (n := index.val) (StatementAbsorption.serializeRunningExpr running)
    (0 : Expr) (Expr.eval env)).symm

/-- Running-transition semantics transport across two interfaces when every
semantic value read by the branch predicate is equal. Complete running-value
equalities imply equality of all canonical serialized words. -/
theorem specHolds_of_cross_values_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (beforeInterface afterInterface : Interface logicalWidth publicFits)
    (beforeOffset afterOffset : Nat) (before after : Env)
    (iterationEq : iterationValue beforeInterface beforeOffset before =
      iterationValue afterInterface afterOffset after)
    (initialStateEq : ∀ index,
      (beforeInterface.initialState beforeOffset index).eval before =
        (afterInterface.initialState afterOffset index).eval after)
    (currentStateEq : ∀ index,
      (beforeInterface.currentState beforeOffset index).eval before =
        (afterInterface.currentState afterOffset index).eval after)
    (recursiveEq : StatementAbsorption.evalRunning
        (beforeInterface.recursive beforeOffset) before =
      StatementAbsorption.evalRunning
        (afterInterface.recursive afterOffset) after)
    (outputEq : StatementAbsorption.evalRunning
        (beforeInterface.output beforeOffset) before =
      StatementAbsorption.evalRunning
        (afterInterface.output afterOffset) after)
    (specification : SpecHolds beforeInterface beforeOffset before) :
    SpecHolds afterInterface afterOffset after := by
  refine {
    initialState := ?_
    base := ?_
    recursive := ?_ }
  · intro afterZero index
    have beforeZero :
        iterationValue beforeInterface beforeOffset before = 0 := by
      rw [iterationEq]
      exact afterZero
    calc
      (afterInterface.initialState afterOffset index).eval after =
          (beforeInterface.initialState beforeOffset index).eval before :=
        (initialStateEq index).symm
      _ = (beforeInterface.currentState beforeOffset index).eval before :=
        specification.initialState beforeZero index
      _ = (afterInterface.currentState afterOffset index).eval after :=
        currentStateEq index
  · intro afterZero index
    have beforeZero :
        iterationValue beforeInterface beforeOffset before = 0 := by
      rw [iterationEq]
      exact afterZero
    have word := specification.base beforeZero index
    rw [runningWord_eval] at word ⊢
    rw [← outputEq]
    exact word
  · intro afterNonzero index
    have beforeNonzero :
        iterationValue beforeInterface beforeOffset before ≠ 0 := by
      intro beforeZero
      apply afterNonzero
      rw [← iterationEq]
      exact beforeZero
    have word := specification.recursive beforeNonzero index
    rw [runningWord_eval, runningWord_eval] at word ⊢
    rw [← outputEq, ← recursiveEq]
    exact word

private theorem serialized_eq_default_of_words {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) (equal : WordsEqualDefault running env) :
    serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning running env) =
      serializeRunning (publicFits := publicFits)
        (defaultRunning (logicalWidth := logicalWidth)
          (publicFits := publicFits)) := by
  apply List.ext_get
  · simp [serializeRunning_length]
  · intro index leftBound rightBound
    have exactBound : index < exactWordCount := by
      simpa [exactWordCount, serializeRunning_length] using leftBound
    have word := equal ⟨index, exactBound⟩
    rw [runningWord_eval] at word
    change _ = (serializeRunning (publicFits := publicFits)
      (defaultRunning (logicalWidth := logicalWidth)
        (publicFits := publicFits))).getD index 0 at word
    rw [List.getD_eq_get _ _ ⟨index, leftBound⟩,
      List.getD_eq_get _ _ ⟨index, rightBound⟩] at word
    exact word

private theorem serialized_eq_running_of_words {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (env : Env) (equal : WordsEqual left right env) :
    serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning left env) =
      serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning right env) := by
  apply List.ext_get
  · simp [serializeRunning_length]
  · intro index leftBound rightBound
    have exactBound : index < exactWordCount := by
      simpa [exactWordCount, serializeRunning_length] using leftBound
    have word := equal ⟨index, exactBound⟩
    rw [runningWord_eval, runningWord_eval] at word
    rw [List.getD_eq_get _ _ ⟨index, leftBound⟩,
      List.getD_eq_get _ _ ⟨index, rightBound⟩] at word
    exact word

/-- The base branch selects the exact canonical serialized default running
instance, including all nonzero framing words. -/
theorem spec_serialized_base {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {interface : Interface logicalWidth publicFits} {offset : Nat}
    {env : Env} (specification : SpecHolds interface offset env)
    (iterationZero : iterationValue interface offset env = 0) :
    serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning (interface.output offset) env) =
      serializeRunning (publicFits := publicFits)
        (defaultRunning (logicalWidth := logicalWidth)
          (publicFits := publicFits)) :=
  serialized_eq_default_of_words _ _ (specification.base iterationZero)

/-- The recursive branch selects every canonical serialized PiDEC output
word, with no digest-only substitution. -/
theorem spec_serialized_recursive {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {interface : Interface logicalWidth publicFits} {offset : Nat}
    {env : Env} (specification : SpecHolds interface offset env)
    (iterationNonzero : iterationValue interface offset env ≠ 0) :
    serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning (interface.output offset) env) =
      serializeRunning (publicFits := publicFits)
        (StatementAbsorption.evalRunning (interface.recursive offset) env) :=
  serialized_eq_running_of_words _ _ _
    (specification.recursive iterationNonzero)

/-- The sole logical circuit for the Stage 1 running-instance branch. -/
def circuit {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => exactPrivateCount
    rowCount := fun _ => exactRowCount
    privateCount_eq := by
      intro offset
      exact localLength_eq interface offset
    rowCount_eq := by
      intro offset
      exact flatConstraints_length_eq interface offset
    soundness := by
      intro env offset assumptions rows
      exact soundness interface env offset assumptions rows
    completeness := by
      intro env offset assumptions specification
      exact completeness interface env offset assumptions specification }

end NightstreamFPrime.Lifecycle.Stage1.RunningTransition
