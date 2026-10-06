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

/-- Support premises for the complete transition interface and its two
logical witnesses: the inverse hint and the stored flag. -/
structure InputsSupported {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop) : Prop where
  iteration : (interface.iteration offset).VarsSatisfy allowed
  inverse : allowed offset
  flag : allowed (offset + 1)
  initialState : ∀ index,
    (interface.initialState offset index).VarsSatisfy allowed
  currentState : ∀ index,
    (interface.currentState offset index).VarsSatisfy allowed
  recursive : RunningSupported (interface.recursive offset) allowed
  output : ∀ index, (interface.output offset index).VarsSatisfy allowed

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

theorem serializeRunningExpr_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : StatementAbsorption.RunningExpr logicalWidth publicFits)
    (allowed : Nat → Prop) (support : RunningSupported running allowed) :
    ∀ expression ∈ StatementAbsorption.serializeRunningExpr running,
      expression.VarsSatisfy allowed := by
  intro expression member
  rcases StatementAbsorption.serializeRunningExpr_mem member with
    ⟨source, row, coefficient, rfl⟩ | ⟨source, coefficient, evalK⟩ |
      ⟨source, matrix, coefficient, evalA⟩ | ⟨coordinate, point⟩ | ⟨word, rfl⟩
  · exact support.commitment source row coefficient
  · exact serializeKExpr_varsSatisfy _ allowed (support.eval_K source coefficient) _ evalK
  · exact serializeKExpr_varsSatisfy _ allowed
      (support.eval_A source matrix coefficient) _ evalA
  · exact serializeKExpr_varsSatisfy _ allowed (support.point coordinate) _ point
  · exact StatementAbsorption.packWordExpr_parent_closed
      (fun expression => expression.VarsSatisfy allowed) (fun _ => trivial)
      (fun left right => Expr.VarsSatisfy.add left right allowed)
      (fun weight value valueSupport =>
        Expr.VarsSatisfy.mul (Expr.const weight) value allowed trivial valueSupport)
      running support.publicInput word

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
    (recursiveFlag interface offset).VarsSatisfy allowed :=
  support.flag

private theorem baseFlag_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (support : InputsSupported interface offset allowed) :
    (baseFlag interface offset).VarsSatisfy allowed :=
  Expr.VarsSatisfy.sub _ _ allowed trivial
    (recursiveFlag_varsSatisfy interface offset allowed support)

/-- Every exact running-transition constraint uses only the selected source
support, the logical inverse witness, and the stored flag. -/
theorem flatConstraints_varsSatisfy {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (support : InputsSupported interface offset allowed) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsSatisfy allowed := by
  have baseFlag := baseFlag_varsSatisfy interface offset allowed support
  intro expression member
  rw [flatConstraints_operations] at member
  simp only [constraints, List.mem_cons, List.mem_append] at member
  rcases member with rfl | rfl | muxMember | stateMember
  · exact Expr.VarsSatisfy.sub _ _ allowed support.flag
      (Expr.VarsSatisfy.mul _ _ allowed support.iteration support.inverse)
  · exact Expr.VarsSatisfy.sub _ _ allowed trivial
      (Expr.VarsSatisfy.mul _ _ allowed support.iteration baseFlag)
  · rcases List.mem_ofFn.mp muxMember with ⟨index, rfl⟩
    exact Expr.VarsSatisfy.sub _ _ allowed
      (Expr.VarsSatisfy.sub _ _ allowed
        (support.output index) trivial)
      (Expr.VarsSatisfy.mul _ _ allowed support.flag
        (Expr.VarsSatisfy.sub _ _ allowed
          (runningWord_varsSatisfy (interface.recursive offset) allowed
            support.recursive index) trivial))
  · rcases List.mem_ofFn.mp stateMember with ⟨index, rfl⟩
    exact Expr.VarsSatisfy.sub _ _ allowed trivial
      (Expr.VarsSatisfy.mul _ _ allowed baseFlag
        (Expr.VarsSatisfy.sub _ _ allowed (support.initialState index)
          (support.currentState index)))

end NightstreamFPrime.Lifecycle.Stage1.RunningTransition
