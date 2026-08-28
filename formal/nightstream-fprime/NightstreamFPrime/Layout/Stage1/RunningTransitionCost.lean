import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Polynomial.Horner

/-!
Owns the exact physical cost of the fixed Stage 1 running transition.
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
      RunningTransition.constraints (interface logicalWidth publicFits)
        phaseOffset := by
  exact RunningTransition.flatConstraints_operations _ _

private theorem binding_directConstraint_eq_none
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.directConstraint
      (RunningTransition.bindingConstraint (interface logicalWidth publicFits)
        phaseOffset) = none := by
  rfl

private theorem binding_mulCount_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.mulCount
      (RunningTransition.bindingConstraint (interface logicalWidth publicFits)
        phaseOffset) = 3 := by
  rfl

private theorem mux_directConstraint_eq_none
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (index : RunningTransition.WordIndex) :
    R1CS.directConstraint
      (RunningTransition.muxConstraint (interface logicalWidth publicFits)
        phaseOffset index) = none := by
  let flag : Expr := iterationExpr * Expr.var phaseOffset
  let base : Expr := 1 - flag
  let recursiveWord : Expr := RunningTransition.runningWord
    (recursiveRunningExpr logicalWidth publicFits) index
  let outputWord : Expr := RunningTransition.runningWord
    (outputRunningExpr logicalWidth publicFits) index
  let selected : Expr :=
    base * Expr.const
        (RunningTransition.defaultWord
          (logicalWidth := logicalWidth) (publicFits := publicFits) index) +
      flag * recursiveWord
  have flagNone : R1CS.lowerAffine flag = none := by
    rfl
  have baseNone : R1CS.lowerAffine base = none := by
    change R1CS.lowerAffine
      (.add (.const 1) (.mul (.const (-1)) flag)) = none
    simp only [R1CS.lowerAffine, flagNone]
  have baseProductNone : R1CS.lowerAffine
      (base * Expr.const
        (RunningTransition.defaultWord
          (logicalWidth := logicalWidth) (publicFits := publicFits) index)) =
      none := by
    change R1CS.lowerAffine (.mul base (.const _)) = none
    rfl
  have selectedNone : R1CS.lowerAffine selected = none := by
    change R1CS.lowerAffine
      (.add
        (base * Expr.const
          (RunningTransition.defaultWord
            (logicalWidth := logicalWidth) (publicFits := publicFits) index))
        (flag * recursiveWord)) = none
    simp only [R1CS.lowerAffine, baseProductNone]
  have wholeNone : R1CS.lowerAffine (selected - outputWord) = none := by
    change R1CS.lowerAffine
      (.add selected (.mul (.const (-1)) outputWord)) = none
    simp only [R1CS.lowerAffine, selectedNone]
  change R1CS.directConstraint (selected - outputWord) = none
  calc
    R1CS.directConstraint (selected - outputWord) =
        R1CS.affineConstraint (selected - outputWord) := by
      rfl
    _ = none := by
      simp [R1CS.affineConstraint, wholeNone]

private theorem mux_mulCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : RunningTransition.WordIndex) :
    R1CS.mulCount
      (RunningTransition.muxConstraint (interface logicalWidth publicFits)
        phaseOffset index) = 6 := by
  have recursiveCount := runningWord_mulCount
    (recursiveRunningExpr logicalWidth publicFits) (recursiveMulFree relation) index
  have outputCount := runningWord_mulCount
    (outputRunningExpr logicalWidth publicFits)
    (outputMulFree logicalWidth publicFits) index
  change R1CS.mulCount
    (((1 - (iterationExpr * Expr.var phaseOffset)) * Expr.const _ +
        (iterationExpr * Expr.var phaseOffset) *
          RunningTransition.runningWord
            (recursiveRunningExpr logicalWidth publicFits) index) -
      RunningTransition.runningWord
        (outputRunningExpr logicalWidth publicFits) index) = 6
  norm_num [R1CS.mulCount, iterationExpr, recursiveCount, outputCount]

private theorem binding_freshCount_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    R1CS.constraintFreshCount
      (RunningTransition.bindingConstraint (interface logicalWidth publicFits)
        phaseOffset) = 3 := by
  rw [R1CS.constraintFreshCount, binding_directConstraint_eq_none,
    binding_mulCount_eq]

private theorem mux_freshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : RunningTransition.WordIndex) :
    R1CS.constraintFreshCount
      (RunningTransition.muxConstraint (interface logicalWidth publicFits)
        phaseOffset index) = 6 := by
  rw [R1CS.constraintFreshCount, mux_directConstraint_eq_none,
    mux_mulCount_eq relation index]

private theorem listSumOfFn {count : Nat} (values : Fin count → Nat) :
    (List.ofFn values).sum = ∑ index, values index := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, Fin.sum_univ_succ, inductionHypothesis]

private theorem totalFreshCount_cons_ofFn {count : Nat}
    (head : Expr) (tail : Fin count → Expr) (headCost tailCost : Nat)
    (headCostEq : R1CS.constraintFreshCount head = headCost)
    (tailCostEq : ∀ index,
      R1CS.constraintFreshCount (tail index) = tailCost) :
    R1CS.totalFreshCount (head :: List.ofFn tail) =
      headCost + count * tailCost := by
  unfold R1CS.totalFreshCount
  rw [List.map_cons, List.sum_cons, List.map_ofFn]
  change R1CS.constraintFreshCount head +
      (List.ofFn fun index : Fin count =>
        R1CS.constraintFreshCount (tail index)).sum =
    headCost + count * tailCost
  rw [headCostEq, listSumOfFn]
  have sumEq :
      (∑ index : Fin count, R1CS.constraintFreshCount (tail index)) =
        ∑ _index : Fin count, tailCost := by
    apply Finset.sum_congr rfl
    intro index _member
    exact tailCostEq index
  rw [sumEq, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  simp

theorem totalFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (logicalConstraints logicalWidth publicFits) =
      275361 := by
  rw [logicalConstraints_eq]
  change R1CS.totalFreshCount
      (RunningTransition.bindingConstraint (interface logicalWidth publicFits)
          phaseOffset ::
        List.ofFn (RunningTransition.muxConstraint
          (interface logicalWidth publicFits) phaseOffset)) = 275361
  calc
    R1CS.totalFreshCount
        (RunningTransition.bindingConstraint (interface logicalWidth publicFits)
            phaseOffset ::
          List.ofFn (RunningTransition.muxConstraint
            (interface logicalWidth publicFits) phaseOffset)) =
        3 + RunningTransition.exactWordCount * 6 :=
      totalFreshCount_cons_ofFn _ _ 3 6
        (binding_freshCount_eq logicalWidth publicFits)
        (mux_freshCount_eq relation)
    _ = 275361 := by rfl

theorem logicalConstraints_length_eq
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (logicalConstraints logicalWidth publicFits).length = 45894 := by
  exact RunningTransition.flatConstraints_length_eq _ _

theorem totalRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (logicalConstraints logicalWidth publicFits) =
      321255 := by
  rw [R1CS.totalRowCount_eq_fresh_add_length,
    totalFreshCount_eq relation, logicalConstraints_length_eq]

end NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
