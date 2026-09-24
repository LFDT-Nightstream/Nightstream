import NightstreamFPrime.Export.Stage1.Wide.RunningSourceValues
import NightstreamFPrime.Layout.Stage1.RunningTransitionReducedRows

/-! Accepted wide physical transition rows imply the reference reduced rows.
Only the inverse and first scratch value are retained from the physical work. -/

namespace NightstreamFPrime.Export.Stage1.Wide.RunningSourceRows

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint Lifecycle.Stage1

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private theorem constraints_transport
    (beforeInterface afterInterface : RunningTransition.Interface width fits)
    (beforeOffset afterOffset : Nat) (before after : Env)
    (iteration : (beforeInterface.iteration beforeOffset).eval before =
      (afterInterface.iteration afterOffset).eval after)
    (inverse : before beforeOffset = after afterOffset)
    (initial : ∀ index, (beforeInterface.initialState beforeOffset index).eval before =
      (afterInterface.initialState afterOffset index).eval after)
    (current : ∀ index, (beforeInterface.currentState beforeOffset index).eval before =
      (afterInterface.currentState afterOffset index).eval after)
    (recursive : ∀ index, (RunningTransition.runningWord (beforeInterface.recursive beforeOffset) index).eval before =
      (RunningTransition.runningWord (afterInterface.recursive afterOffset) index).eval after)
    (output : ∀ index, (RunningTransition.runningWord (beforeInterface.output beforeOffset) index).eval before =
      (RunningTransition.runningWord (afterInterface.output afterOffset) index).eval after)
    (holds : ConstraintsHold after (RunningTransition.constraints afterInterface afterOffset)) :
    ConstraintsHold before (RunningTransition.constraints beforeInterface beforeOffset) := by
  intro expression member
  simp only [RunningTransition.constraints, RunningTransition.muxConstraints,
    RunningTransition.baseStateConstraints, List.mem_cons, List.mem_append, List.mem_ofFn] at member
  rcases member with rfl | ⟨index, rfl⟩ | ⟨index, rfl⟩
  · have valid := holds (RunningTransition.bindingConstraint afterInterface afterOffset) List.mem_cons_self
    simpa only [RunningTransition.bindingConstraint, RunningTransition.baseFlag,
      RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
      Expr.eval_sub, Expr.eval_hmul, Expr.eval_const, Expr.eval_var, iteration, inverse] using valid
  · have valid := holds (RunningTransition.muxConstraint afterInterface afterOffset index)
      (by simp only [RunningTransition.constraints, RunningTransition.muxConstraints,
        List.mem_cons, List.mem_append, List.mem_ofFn]; exact Or.inr (Or.inl ⟨index, rfl⟩))
    simpa only [RunningTransition.muxConstraint, RunningTransition.baseFlag,
      RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
      Expr.eval_sub, Expr.eval_hmul, Expr.eval_hadd, Expr.eval_const, Expr.eval_var,
      iteration, inverse, recursive, output] using valid
  · have valid := holds (RunningTransition.baseStateConstraint afterInterface afterOffset index)
      (by simp only [RunningTransition.constraints, RunningTransition.baseStateConstraints,
        List.mem_cons, List.mem_append, List.mem_ofFn]; exact Or.inr (Or.inr ⟨index, rfl⟩))
    simpa only [RunningTransition.baseStateConstraint, RunningTransition.baseFlag,
      RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
      Expr.eval_sub, Expr.eval_hmul, Expr.eval_const, Expr.eval_var,
      iteration, inverse, initial, current] using valid

theorem inverse (env : Env) :
    SourceAssignment.sourceEnv env Layout.Stage1.RunningTransitionInputs.phaseOffset =
      env Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset :=
  RunningSourceValues.suffix env 67338 (by decide)

theorem flag (env : Env) :
    SourceAssignment.sourceEnv env Layout.Stage1.RunningTransitionReducedRows.flagIndex =
      env Layout.Stage1.Wide.RunningTransitionLayout.logicalColumnCount :=
  RunningSourceValues.suffix env 67339 (by decide)

/-- The first multiplication row binds the wide shared flag. -/
theorem physical_flag (env : Env)
    (physical : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    env Layout.Stage1.Wide.RunningTransitionLayout.logicalColumnCount =
      (Layout.Stage1.Wide.RunningTransitionInputs.iterationExpr).eval env *
        env Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset := by
  let row : R1CS.Row := ⟨R1CS.LinearCombination.ofVar 28,
    R1CS.LinearCombination.ofVar Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset,
    R1CS.LinearCombination.ofVar Layout.Stage1.Wide.RunningTransitionLayout.logicalColumnCount⟩
  have valid := physical row (by
    rw [Layout.Stage1.Wide.RunningTransitionLayout.physicalRows, R1CS.LoweringPlan.rows,
      R1CS.LoweringPlan.lowering, Layout.Stage1.Wide.RunningTransitionLayout.plan_constraints,
      Layout.Stage1.Wide.RunningTransitionLayout.plan_firstFresh,
      Layout.Stage1.Wide.RunningTransitionLayout.logicalConstraints_eq, RunningTransition.constraints]
    apply List.mem_append_left
    change row ∈ row :: _
    exact List.mem_cons_self)
  symm
  simpa only [row, R1CS.Row.Holds, R1CS.LinearCombination.eval_ofVar,
    Layout.Stage1.Wide.RunningTransitionInputs.iterationExpr, Expr.eval_var] using valid

/-- The reference source view satisfies the same 49,359 reduced equations. -/
theorem reduced_rows (relation : ProductionKey.LogicalRelation width fits) (env : Env)
    (physical : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    R1CS.RowsHold (SourceAssignment.sourceEnv env) (Layout.Stage1.RunningTransitionReducedRows.rows width fits) := by
  apply (Layout.Stage1.RunningTransitionReducedRows.rows_iff_logical relation _).mpr
  constructor
  · rw [flag, inverse]
    exact (physical_flag env physical).trans (congrArg (fun value => value * env
      Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset)
      (RunningSourceValues.iteration (width := width) (fits := fits) env).symm)
  · have logical := R1CS.LoweringPlan.sound (Layout.Stage1.Wide.RunningTransitionLayout.plan width fits) env physical
    rw [Layout.Stage1.Wide.RunningTransitionLayout.plan_constraints,
      Layout.Stage1.Wide.RunningTransitionLayout.logicalConstraints_eq] at logical
    rw [Layout.Stage1.RunningTransitionLayout.logicalConstraints_eq]
    exact constraints_transport _ _ _ _ _ _ (RunningSourceValues.iteration env) (inverse env)
      (RunningSourceValues.initial env) (RunningSourceValues.current env)
      (RunningSourceValues.recursive_word env) (RunningSourceValues.output_word env) logical

end NightstreamFPrime.Export.Stage1.Wide.RunningSourceRows
