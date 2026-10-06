import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Polynomial.Horner

/-!
Owns the exact R1CS footprint and endpoints of the Stage 1 running transition.
The proof is structural and does not normalize the complete running vector.
-/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def logicalColumnCount : Nat :=
  phaseOffset + RunningTransition.exactPrivateCount

/-- Every running-transition constraint is one rank-one row, so the
transition allocates no fresh column. -/
def exactFreshCount : Nat := 0

/-- The running transition ends after its logical and R1CS fresh columns. -/
def physicalEnd : Nat := logicalColumnCount + exactFreshCount

structure RunningMulFree {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits) : Prop where
  point : ∀ coordinate, KExprLinear (running.point coordinate)
  commitment : ∀ source row coefficient,
    R1CS.mulCount (running.commitment source row coefficient) = 0
  publicInput : ∀ source column,
    R1CS.mulCount (running.publicInput source column) = 0
  eval_K : ∀ source coefficient,
    KExprLinear ((running.evaluation source).eval_K coefficient)
  eval_A : ∀ source matrix coefficient,
    KExprLinear ((running.evaluation source).eval_A matrix coefficient)

private theorem serializeKExpr_mulFree (value : KExpr)
    (linear : KExprLinear value) :
    ∀ expression ∈ StatementAbsorption.serializeKExpr value,
      R1CS.mulCount expression = 0 := by
  intro expression member
  simp only [StatementAbsorption.serializeKExpr, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact linear.c0_mulCount
  · exact linear.c1_mulCount

private theorem serializePointExpr_mulFree
    (point : Fin productionShape.cubeVariables → KExpr)
    (linear : ∀ coordinate, KExprLinear (point coordinate)) :
    ∀ expression ∈ StatementAbsorption.serializePointExpr point,
      R1CS.mulCount expression = 0 := by
  intro expression member
  rw [StatementAbsorption.serializePointExpr, List.mem_flatMap] at member
  rcases member with ⟨coordinate, _coordinateMember, expressionMember⟩
  exact serializeKExpr_mulFree (point coordinate) (linear coordinate)
    expression expressionMember

private theorem serializeCommitmentExpr_mulFree
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr)
    (linear : ∀ row coefficient,
      R1CS.mulCount (commitment row coefficient) = 0) :
    ∀ expression ∈ StatementAbsorption.serializeCommitmentExpr commitment,
      R1CS.mulCount expression = 0 := by
  intro expression member
  rw [StatementAbsorption.serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _rowMember, expressionMember⟩
  rw [List.mem_map] at expressionMember
  rcases expressionMember with ⟨coefficient, _coefficientMember, rfl⟩
  exact linear row coefficient

private theorem serializePublicInputExpr_mulFree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (linear : ∀ column, R1CS.mulCount (input column) = 0) :
    ∀ expression ∈ StatementAbsorption.serializePublicInputExpr input,
      R1CS.mulCount expression = 0 := by
  intro expression member
  rw [StatementAbsorption.serializePublicInputExpr, List.mem_map] at member
  rcases member with ⟨column, _columnMember, rfl⟩
  exact linear column

private theorem serializeEvaluationExpr_mulFree
    (evaluation : StatementAbsorption.EvaluationExpr)
    (eval_K : ∀ coefficient, KExprLinear (evaluation.eval_K coefficient))
    (eval_A : ∀ matrix coefficient,
      KExprLinear (evaluation.eval_A matrix coefficient)) :
    ∀ expression ∈ StatementAbsorption.serializeEvaluationExpr evaluation,
      R1CS.mulCount expression = 0 := by
  intro expression member
  rw [StatementAbsorption.serializeEvaluationExpr, List.mem_append] at member
  rcases member with padMember | matrixMember
  · rw [List.mem_flatMap] at padMember
    rcases padMember with ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_mulFree (evaluation.eval_K coefficient)
      (eval_K coefficient) expression expressionMember
  · rw [List.mem_flatMap] at matrixMember
    rcases matrixMember with ⟨matrix, _matrixMember, coefficientMember⟩
    rw [List.mem_flatMap] at coefficientMember
    rcases coefficientMember with
      ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_mulFree (evaluation.eval_A matrix coefficient)
      (eval_A matrix coefficient) expression expressionMember

private theorem blockExpr_mulFree (words : List Expr)
    (linear : ∀ expression ∈ words, R1CS.mulCount expression = 0) :
    ∀ expression ∈ StatementAbsorption.blockExpr words,
      R1CS.mulCount expression = 0 := by
  intro expression member
  simp only [StatementAbsorption.blockExpr, List.mem_cons] at member
  rcases member with rfl | wordMember
  · rfl
  · exact linear expression wordMember

private theorem serializeRunningExpr_mulFree {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) :
    ∀ expression ∈ StatementAbsorption.serializeRunningExpr running,
      R1CS.mulCount expression = 0 := by
  intro expression member
  rw [StatementAbsorption.serializeRunningExpr, List.mem_append] at member
  rcases member with pointMember | groupMember
  · exact blockExpr_mulFree _
      (serializePointExpr_mulFree running.point linear.point)
      expression pointMember
  · rw [List.mem_flatMap] at groupMember
    rcases groupMember with ⟨source, _sourceMember, expressionMember⟩
    simp only [List.mem_append] at expressionMember
    rcases expressionMember with (commitmentMember | publicMember) |
      evaluationMember
    · exact blockExpr_mulFree _
        (serializeCommitmentExpr_mulFree (running.commitment source)
          (linear.commitment source)) expression commitmentMember
    · exact blockExpr_mulFree _
        (serializePublicInputExpr_mulFree (running.publicInput source)
          (linear.publicInput source)) expression publicMember
    · exact blockExpr_mulFree _
        (serializeEvaluationExpr_mulFree (running.evaluation source)
          (linear.eval_K source) (linear.eval_A source))
        expression evaluationMember

theorem runningWord_mulCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) (index : RunningTransition.WordIndex) :
    R1CS.mulCount (RunningTransition.runningWord running index) = 0 := by
  have indexBound : index.val <
      (StatementAbsorption.serializeRunningExpr running).length := by
    rw [StatementAbsorption.serializeRunningExpr_length]
    exact index.isLt
  rw [RunningTransition.runningWord,
    List.getD_eq_get _ _ ⟨index.val, indexBound⟩]
  exact serializeRunningExpr_mulFree running linear _
    (List.get_mem _ ⟨index.val, indexBound⟩)

def outputMulFree
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningMulFree (outputRunningExpr logicalWidth publicFits) := by
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [outputRunningExpr, outputPoint, outputPairAt, Nonconstant]
  · intro source row coefficient
    rfl
  · intro source column
    rfl
  · intro source coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [outputRunningExpr, outputEval_K, outputPairAt, Nonconstant]
  · intro source matrix coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [outputRunningExpr, outputEval_A, outputPairAt, Nonconstant]

def recursiveMulFree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    RunningMulFree (recursiveRunningExpr logicalWidth publicFits) := by
  refine {
    point := recursivePointLinear relation
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro source row coefficient
    rfl
  · intro source column
    rfl
  · intro source coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
        PiDECInputs.message, PiDECInputs.childEvalK, Nonconstant]
  · intro source matrix coefficient
    refine ⟨rfl, rfl, ?_, ?_⟩ <;>
      simp [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
        PiDECInputs.message, PiDECInputs.childEvalA, Nonconstant]

def logicalConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  flatConstraints
    (RunningTransition.operations (interface logicalWidth publicFits) phaseOffset)

theorem logicalConstraints_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    logicalConstraints logicalWidth publicFits =
      RunningTransition.flagConstraint (interface logicalWidth publicFits)
          phaseOffset ::
        RunningTransition.constraints (interface logicalWidth publicFits)
          phaseOffset := by
  exact RunningTransition.flatConstraints_operations _ _

private theorem runningWord_affine {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (linear : RunningMulFree running) (index : RunningTransition.WordIndex) :
    R1CS.IsAffine
      (RunningTransition.runningWord running index - Expr.const
        (RunningTransition.defaultWord
          (logicalWidth := logicalWidth) (publicFits := publicFits) index)) :=
  R1CS.IsAffine.add
    (isAffine_of_mulCount_zero _ (runningWord_mulCount running linear index))
    (R1CS.IsAffine.const_mul _ (R1CS.isAffine_const _))

private theorem baseFlag_affine
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.IsAffine
      (RunningTransition.baseFlag (interface logicalWidth publicFits)
        phaseOffset) :=
  R1CS.IsAffine.add (R1CS.isAffine_const _)
    (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))

/-- The flag recipe and every assertion are one rank-one row. -/
private theorem constraintFreshCount_eq_zero
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ expression ∈ logicalConstraints logicalWidth publicFits,
      R1CS.constraintFreshCount expression = 0 := by
  have iterationAffine : R1CS.IsAffine iterationExpr := R1CS.isAffine_var _
  intro expression member
  rw [logicalConstraints_eq] at member
  simp only [RunningTransition.constraints, List.mem_cons,
    List.mem_append] at member
  rcases member with rfl | rfl | muxMember | stateMember
  · exact R1CS.constraintFreshCount_recipe_eq_zero _ _
      (R1CS.IsDirectRecipe.mul _ iterationAffine (R1CS.isAffine_var _))
  · exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (R1CS.isAffine_const _) iterationAffine
      (baseFlag_affine logicalWidth publicFits)
  · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
    exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (runningWord_affine _ (outputMulFree logicalWidth publicFits) index)
      (R1CS.isAffine_var _)
      (runningWord_affine _ (recursiveMulFree relation) index)
  · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
    exact R1CS.constraintFreshCount_rankOne_eq_zero _ _ _
      (R1CS.isAffine_const _) (baseFlag_affine logicalWidth publicFits)
      (R1CS.IsAffine.add (R1CS.isAffine_var _)
        (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _)))

/-- Structural lowering uses exactly the fresh columns declared by the footprint. -/
theorem totalFreshCount_eq_exactFreshCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (logicalConstraints logicalWidth publicFits) =
      exactFreshCount :=
  R1CS.totalFreshCount_eq_zero_of_noFresh _
    (constraintFreshCount_eq_zero relation)

theorem totalFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (logicalConstraints logicalWidth publicFits) = 0 :=
  totalFreshCount_eq_exactFreshCount relation

theorem logicalConstraints_length_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (logicalConstraints logicalWidth publicFits).length = 37261 := by
  exact RunningTransition.flatConstraints_length_eq _ _

theorem totalRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (logicalConstraints logicalWidth publicFits) =
      37261 := by
  rw [R1CS.totalRowCount_eq_fresh_add_length,
    totalFreshCount_eq relation, logicalConstraints_length_eq]

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
