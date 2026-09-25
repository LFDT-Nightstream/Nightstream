import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.Stage1.RunningTransition

/-!
Owns generic variable-support propagation for the Stage 1 running
transition. It does not select physical columns or a retained assignment.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.RunningTransition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Field-level support for one complete symbolic running vector. -/
structure RunningSupported {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (allowed : Nat → Prop) : Prop where
  point : ∀ coordinate,
    (running.point coordinate).c0.VarsSatisfy allowed ∧
      (running.point coordinate).c1.VarsSatisfy allowed
  commitment : ∀ source row coefficient,
    (running.commitment source row coefficient).VarsSatisfy allowed
  publicInput : ∀ source column,
    (running.publicInput source column).VarsSatisfy allowed
  eval_K : ∀ source coefficient,
    ((running.evaluation source).eval_K coefficient).c0.VarsSatisfy allowed ∧
      ((running.evaluation source).eval_K coefficient).c1.VarsSatisfy allowed
  eval_A : ∀ source matrix coefficient,
    ((running.evaluation source).eval_A matrix coefficient).c0.VarsSatisfy
        allowed ∧
      ((running.evaluation source).eval_A matrix coefficient).c1.VarsSatisfy
        allowed

/-- Support premises for the complete transition interface and its sole
logical witness. -/
structure InputsSupported {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop) : Prop where
  iteration : (interface.iteration offset).VarsSatisfy allowed
  inverse : allowed offset
  initialState : ∀ index,
    (interface.initialState offset index).VarsSatisfy allowed
  currentState : ∀ index,
    (interface.currentState offset index).VarsSatisfy allowed
  recursive : RunningSupported (interface.recursive offset) allowed
  output : RunningSupported (interface.output offset) allowed

private theorem serializeKExpr_varsSatisfy (value : KExpr)
    (allowed : Nat → Prop)
    (support : value.c0.VarsSatisfy allowed ∧
      value.c1.VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.serializeKExpr value,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [StatementAbsorption.serializeKExpr, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact support.1
  · exact support.2

private theorem serializePointExpr_varsSatisfy
    (point : Fin productionShape.cubeVariables → KExpr)
    (allowed : Nat → Prop)
    (support : ∀ coordinate,
      (point coordinate).c0.VarsSatisfy allowed ∧
        (point coordinate).c1.VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.serializePointExpr point,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StatementAbsorption.serializePointExpr, List.mem_flatMap] at member
  rcases member with ⟨coordinate, _coordinateMember, expressionMember⟩
  exact serializeKExpr_varsSatisfy (point coordinate) allowed
    (support coordinate) expression expressionMember

private theorem serializeCommitmentExpr_varsSatisfy
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) (allowed : Nat → Prop)
    (support : ∀ row coefficient,
      (commitment row coefficient).VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.serializeCommitmentExpr commitment,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StatementAbsorption.serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _rowMember, expressionMember⟩
  rw [List.mem_map] at expressionMember
  rcases expressionMember with
    ⟨coefficient, _coefficientMember, rfl⟩
  exact support row coefficient

private theorem serializePublicInputExpr_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (allowed : Nat → Prop)
    (support : ∀ column, (input column).VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.serializePublicInputExpr input,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StatementAbsorption.serializePublicInputExpr, List.mem_map] at member
  rcases member with ⟨column, _columnMember, rfl⟩
  exact support column

private theorem serializeEvaluationExpr_varsSatisfy
    (evaluation : StatementAbsorption.EvaluationExpr)
    (allowed : Nat → Prop)
    (eval_K : ∀ coefficient,
      (evaluation.eval_K coefficient).c0.VarsSatisfy allowed ∧
        (evaluation.eval_K coefficient).c1.VarsSatisfy allowed)
    (eval_A : ∀ matrix coefficient,
      (evaluation.eval_A matrix coefficient).c0.VarsSatisfy allowed ∧
        (evaluation.eval_A matrix coefficient).c1.VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.serializeEvaluationExpr evaluation,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StatementAbsorption.serializeEvaluationExpr, List.mem_append] at member
  rcases member with padMember | matrixMember
  · rw [List.mem_flatMap] at padMember
    rcases padMember with
      ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_varsSatisfy (evaluation.eval_K coefficient) allowed
      (eval_K coefficient) expression expressionMember
  · rw [List.mem_flatMap] at matrixMember
    rcases matrixMember with
      ⟨matrix, _matrixMember, coefficientMember⟩
    rw [List.mem_flatMap] at coefficientMember
    rcases coefficientMember with
      ⟨coefficient, _coefficientMember, expressionMember⟩
    exact serializeKExpr_varsSatisfy
      (evaluation.eval_A matrix coefficient) allowed
      (eval_A matrix coefficient) expression expressionMember

private theorem blockExpr_varsSatisfy (words : List Expr)
    (allowed : Nat → Prop)
    (support : ∀ expression ∈ words,
      expression.VarsSatisfy allowed) :
    ∀ expression ∈ StatementAbsorption.blockExpr words,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [StatementAbsorption.blockExpr, List.mem_cons] at member
  rcases member with rfl | wordMember
  · trivial
  · exact support expression wordMember

theorem serializeRunningExpr_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (allowed : Nat → Prop) (support : RunningSupported running allowed) :
    ∀ expression ∈ StatementAbsorption.serializeRunningExpr running,
      expression.VarsSatisfy allowed := by
  intro expression member
  rw [StatementAbsorption.serializeRunningExpr, List.mem_append] at member
  rcases member with pointMember | groupMember
  · exact blockExpr_varsSatisfy _ allowed
      (serializePointExpr_varsSatisfy running.point allowed support.point)
      expression pointMember
  · rw [List.mem_flatMap] at groupMember
    rcases groupMember with ⟨source, _sourceMember, expressionMember⟩
    simp only [List.mem_append] at expressionMember
    rcases expressionMember with (commitmentMember | publicMember) |
      evaluationMember
    · exact blockExpr_varsSatisfy _ allowed
        (serializeCommitmentExpr_varsSatisfy
          (running.commitment source) allowed (support.commitment source))
        expression commitmentMember
    · exact blockExpr_varsSatisfy _ allowed
        (serializePublicInputExpr_varsSatisfy
          (running.publicInput source) allowed (support.publicInput source))
        expression publicMember
    · exact blockExpr_varsSatisfy _ allowed
        (serializeEvaluationExpr_varsSatisfy (running.evaluation source)
          allowed (support.eval_K source) (support.eval_A source))
        expression evaluationMember

theorem runningWord_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (allowed : Nat → Prop) (support : RunningSupported running allowed)
    (index : WordIndex) :
    (runningWord running index).VarsSatisfy allowed := by
  have indexBound : index.val <
      (StatementAbsorption.serializeRunningExpr running).length := by
    rw [StatementAbsorption.serializeRunningExpr_length]
    exact index.isLt
  rw [runningWord, List.getD_eq_get _ _ ⟨index.val, indexBound⟩]
  exact serializeRunningExpr_varsSatisfy running allowed support _
    (List.get_mem _ ⟨index.val, indexBound⟩)

private theorem recursiveFlag_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (support : InputsSupported interface offset allowed) :
    (recursiveFlag interface offset).VarsSatisfy allowed := by
  exact ⟨support.iteration, support.inverse⟩

private theorem baseFlag_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (support : InputsSupported interface offset allowed) :
    (baseFlag interface offset).VarsSatisfy allowed := by
  exact ⟨trivial, ⟨trivial,
    recursiveFlag_varsSatisfy interface offset allowed support⟩⟩

/-- Every exact running-transition constraint uses only the selected source
support and the sole logical inverse witness selected by that support. -/
theorem constraints_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (support : InputsSupported interface offset allowed) :
    ∀ expression ∈ constraints interface offset,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [constraints, List.mem_cons, List.mem_append] at member
  rcases member with rfl | muxMember | stateMember
  · exact ⟨support.iteration,
      baseFlag_varsSatisfy interface offset allowed support⟩
  · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
    change
      (((baseFlag interface offset).VarsSatisfy allowed ∧ True) ∧
        ((recursiveFlag interface offset).VarsSatisfy allowed ∧
          (runningWord (interface.recursive offset) index).VarsSatisfy
            allowed)) ∧
        (True ∧ (runningWord (interface.output offset) index).VarsSatisfy
          allowed)
    exact ⟨⟨⟨baseFlag_varsSatisfy interface offset allowed support,
          trivial⟩,
        ⟨recursiveFlag_varsSatisfy interface offset allowed support,
          runningWord_varsSatisfy (interface.recursive offset) allowed
            support.recursive index⟩⟩,
      ⟨trivial,
        runningWord_varsSatisfy (interface.output offset) allowed
          support.output index⟩⟩
  · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
    simp only [baseStateConstraint, Expr.VarsSatisfy]
    exact ⟨baseFlag_varsSatisfy interface offset allowed support,
      support.initialState index, trivial, support.currentState index⟩

end NightstreamFPrime.Lifecycle.Stage1.RunningTransition
