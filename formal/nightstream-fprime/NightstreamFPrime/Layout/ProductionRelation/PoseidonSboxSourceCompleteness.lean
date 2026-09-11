import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxPlan

/-!
Owns the adjacent conversion from satisfying canonical permutation recipe
rows to the compact compiler's retained S-box equations. The source trace,
schedule, and retained-output order are the existing owners' definitions.
The proof composes one schedule step at a time and never enumerates package
invocations or evaluates a row family.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PoseidonSboxPlan

private def sourceOutputs (start : Nat) (state : Permutation.EState)
    (steps : List Permutation.Step) : List Expr :=
  (PoseidonScheduleTrace.recordsFrom start state steps).flatMap fun record =>
    (PoseidonStepTrace.sboxProgram record.start record.step record.state).outputs

private theorem sourceOutputs_cons (start : Nat) (state : Permutation.EState)
    (step : Permutation.Step) (rest : List Permutation.Step) :
    sourceOutputs start state (step :: rest) =
      (PoseidonStepTrace.sboxProgram start step state).outputs ++
        sourceOutputs (start + Permutation.stepSize step)
          (Permutation.stepOutput start step) rest := by
  rfl

private theorem mapped_sbox_value (env : Env) (inputs outputs : List Expr)
    (equal : outputs.map (Expr.eval env) =
      inputs.map (fun input => Layer.sboxF (input.eval env))) (index : Nat) :
    (outputs.getD index 0).eval env =
      Layer.sboxF ((inputs.getD index 0).eval env) := by
  have observed := congrArg (fun values : List F => values.getD index 0) equal
  have left := List.getD_map (l := outputs) (d := (0 : Expr))
    (n := index) (Expr.eval env)
  have right := List.getD_map (l := inputs) (d := (0 : Expr))
    (n := index) (fun input => Layer.sboxF (input.eval env))
  have zero : Layer.sboxF ((0 : Expr).eval env) = 0 := by
    change Layer.sboxF (0 : F) = 0
    simp [Layer.sboxF, Poseidon2.sbox]
  change (outputs.map (Expr.eval env)).getD index 0 = _ at left
  rw [zero] at right
  exact left.symm.trans (observed.trans right)

private theorem full_equations
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (env : Env)
    (one : assignment interface.oneColumn = 1)
    (constants : List (List Nat)) (round nextSbox start : Nat)
    (state : State logicalWidth) (sourceState : Permutation.EState)
    (inputEq : SparseLayer.evalState assignment state =
      Layer.evalState env sourceState)
    (outputs : ∀ lane : Fin 8,
      (fullOutput interface nextSbox lane).eval assignment =
        (Permutation.fullSboxState start constants round sourceState lane).eval env)
    (trace : (Permutation.compileSboxes start
        (Permutation.fullInputs constants round sourceState)).outputs.map (Expr.eval env) =
      (Permutation.fullInputs constants round sourceState).map
        (fun input => Layer.sboxF (input.eval env))) :
    ∀ forms ∈ fullRows interface constants round nextSbox state,
      forms.output.eval assignment = Layer.sboxF (forms.input.eval assignment) := by
  intro forms member
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
  change (fullOutput interface nextSbox lane).eval assignment = _
  rw [outputs]
  have sourceEq := mapped_sbox_value env _ _ trace lane.val
  change (Permutation.fullSboxState start constants round sourceState lane).eval env = _
    at sourceEq
  rw [sourceEq]
  apply congrArg Layer.sboxF
  rw [fullInput, SparseLayer.eval_addConstant assignment interface.oneColumn one]
  have valueEq := congrFun inputEq lane
  change (state lane).eval assignment = (sourceState lane).eval env at valueEq
  rw [valueEq]
  have laneBound : lane.val < (Permutation.fullInputs constants round sourceState).length := by
    simpa only [Permutation.fullInputs, List.length_ofFn] using lane.isLt
  rw [List.getD_eq_getElem (l := Permutation.fullInputs constants round sourceState)
    (d := (0 : Expr)) laneBound]
  simp only [Permutation.fullInputs, List.getElem_ofFn, Expr.eval_hadd, Expr.eval_const]

private theorem step_equations
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (env : Env)
    (one : assignment interface.oneColumn = 1)
    (nextSbox start : Nat) (state : State logicalWidth)
    (sourceState : Permutation.EState) (step : Permutation.Step)
    (inputEq : SparseLayer.evalState assignment state =
      Layer.evalState env sourceState)
    (outputs : ∀ index,
      index < (PoseidonStepTrace.sboxProgram start step sourceState).outputs.length →
      (sboxOutputAt interface (nextSbox + index)).eval assignment =
        ((PoseidonStepTrace.sboxProgram start step sourceState).outputs.getD index 0).eval env)
    (rows : ConstraintsHold env
      (recipeConstraints start (Permutation.stepRecipes start step sourceState))) :
    ∀ forms ∈ (compileStep interface nextSbox state step).rows,
      forms.output.eval assignment = Layer.sboxF (forms.input.eval assignment) := by
  have trace := (PoseidonStepTrace.rows_imply_holds env start step sourceState rows).1
  cases step with
  | initialLayer => simp [compileStep]
  | initialFullRound round =>
      apply full_equations interface assignment env one Poseidon2.initialConstants
        round nextSbox start state sourceState inputEq _ trace
      intro lane
      apply outputs lane.val
      simpa only [PoseidonStepTrace.sboxProgram, PoseidonStepTrace.sboxInputs,
        Permutation.compileSboxes_outputs_length, Permutation.fullInputs,
        List.length_ofFn] using lane.isLt
  | terminalFullRound round =>
      apply full_equations interface assignment env one Poseidon2.terminalConstants
        round nextSbox start state sourceState inputEq _ trace
      intro lane
      apply outputs lane.val
      simpa only [PoseidonStepTrace.sboxProgram, PoseidonStepTrace.sboxInputs,
        Permutation.compileSboxes_outputs_length, Permutation.fullInputs,
        List.length_ofFn] using lane.isLt
  | partialRound round =>
      intro forms member
      have formsEq : forms =
          { selector := selector interface
            input := partialInput interface round state
            output := partialOutput interface nextSbox } := by
        simpa [compileStep, partialRows] using member
      subst forms
      have outputEq := outputs 0 (by
        simp [PoseidonStepTrace.sboxProgram, PoseidonStepTrace.sboxInputs])
      have sourceEq := mapped_sbox_value env _ _ trace 0
      change (partialOutput interface nextSbox).eval assignment = _
      rw [partialOutput, ← Nat.add_zero nextSbox, outputEq, sourceEq]
      apply congrArg Layer.sboxF
      rw [partialInput, SparseLayer.eval_addConstant assignment interface.oneColumn one]
      have stateEq := congrFun inputEq 0
      change (state 0).eval assignment = (sourceState 0).eval env at stateEq
      rw [stateEq]
      simp [PoseidonStepTrace.sboxInputs, Permutation.partialInput]

private theorem step_rowsZero
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (nextSbox : Nat)
    (state : State logicalWidth) (step : Permutation.Step)
    (equations : ∀ forms ∈ (compileStep interface nextSbox state step).rows,
      forms.output.eval assignment = Layer.sboxF (forms.input.eval assignment)) :
    SboxRowsZero assignment (compileStep interface nextSbox state step).rows := by
  intro forms member
  rw [SboxRow.Forms.residual_eq, equations forms member]
  have power : Spec.ProductionRelation.RowSemantics.seventhPower
      (forms.input.eval assignment) = Layer.sboxF (forms.input.eval assignment) := by
    simp [Spec.ProductionRelation.RowSemantics.seventhPower,
      Spec.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
      Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
      Layer.sboxF, Poseidon2.sbox, mul_assoc]
  rw [power, sub_self, mul_zero]

private theorem step_nextSbox
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (nextSbox start : Nat) (state : State logicalWidth)
    (sourceState : Permutation.EState) (step : Permutation.Step) :
    (compileStep interface nextSbox state step).nextSbox = nextSbox +
      (PoseidonStepTrace.sboxProgram start step sourceState).outputs.length := by
  cases step <;>
    simp [compileStep, PoseidonStepTrace.sboxProgram, PoseidonStepTrace.sboxInputs,
      Permutation.fullInputs]

private theorem compile_equations
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (env : Env)
    (one : assignment interface.oneColumn = 1)
    (steps : List Permutation.Step) (nextSbox start : Nat)
    (state : State logicalWidth) (sourceState : Permutation.EState)
    (inputEq : SparseLayer.evalState assignment state = Layer.evalState env sourceState)
    (outputs : ∀ index, index < (sourceOutputs start sourceState steps).length →
      (sboxOutputAt interface (nextSbox + index)).eval assignment =
        ((sourceOutputs start sourceState steps).getD index 0).eval env)
    (rows : ConstraintsHold env (recipeConstraints start
      (Permutation.compile start sourceState steps).recipes)) :
    ∀ forms ∈ (compile interface nextSbox state steps).rows,
      forms.output.eval assignment = Layer.sboxF (forms.input.eval assignment) := by
  induction steps generalizing nextSbox start state sourceState with
  | nil => simp [compile]
  | cons step rest induction =>
      have separated :
          ConstraintsHold env (recipeConstraints start
            (Permutation.stepRecipes start step sourceState)) ∧
          ConstraintsHold env (recipeConstraints (start + Permutation.stepSize step)
            (Permutation.compile (start + Permutation.stepSize step)
              (Permutation.stepOutput start step) rest).recipes) := by
        rw [Permutation.compile, Permutation.recipeConstraints_append] at rows
        simpa using (constraintsHold_append env _ _).mp rows
      have headOutputs : ∀ index,
          index < (PoseidonStepTrace.sboxProgram start step sourceState).outputs.length →
          (sboxOutputAt interface (nextSbox + index)).eval assignment =
            ((PoseidonStepTrace.sboxProgram start step sourceState).outputs.getD index 0).eval env := by
        intro index bound
        have outputEq := outputs index (by rw [sourceOutputs_cons, List.length_append]; omega)
        rw [sourceOutputs_cons, List.getD_append _ _ _ _ bound] at outputEq
        exact outputEq
      have head := step_equations interface assignment env one nextSbox start
        state sourceState step inputEq headOutputs separated.1
      have stateEq : SparseLayer.evalState assignment
          (compileStep interface nextSbox state step).state =
          Layer.evalState env (Permutation.stepOutput start step) := by
        rw [compileStep_sound interface nextSbox state step assignment one
          (step_rowsZero interface assignment nextSbox state step head), inputEq]
        exact (Permutation.stepRows_sound env start step sourceState separated.1).symm
      have tailOutputs : ∀ index,
          index < (sourceOutputs (start + Permutation.stepSize step)
            (Permutation.stepOutput start step) rest).length →
          (sboxOutputAt interface
            ((compileStep interface nextSbox state step).nextSbox + index)).eval assignment =
          ((sourceOutputs (start + Permutation.stepSize step)
            (Permutation.stepOutput start step) rest).getD index 0).eval env := by
        intro index bound
        have outputEq := outputs
          ((PoseidonStepTrace.sboxProgram start step sourceState).outputs.length + index)
          (by rw [sourceOutputs_cons, List.length_append]; omega)
        rw [sourceOutputs_cons, List.getD_append_right _ _ _ _ (by omega),
          Nat.add_sub_cancel_left] at outputEq
        rw [step_nextSbox interface nextSbox start state sourceState step, Nat.add_assoc]
        exact outputEq
      have tail := induction
        (compileStep interface nextSbox state step).nextSbox
        (start + Permutation.stepSize step)
        (compileStep interface nextSbox state step).state
        (Permutation.stepOutput start step) stateEq tailOutputs separated.2
      intro forms member
      change forms ∈ (compileStep interface nextSbox state step).rows ++
        (compile interface (compileStep interface nextSbox state step).nextSbox
          (compileStep interface nextSbox state step).state rest).rows at member
      exact (List.mem_append.mp member).elim (head forms) (tail forms)

private theorem retained_outputs_eq :
    PoseidonRetainedSlots.rows.map (fun row => row.outputExpression) =
      sourceOutputs PoseidonScheduleTrace.inputCount
        PoseidonScheduleTrace.canonicalState Permutation.schedule := by
  unfold PoseidonRetainedSlots.rows
  rw [List.map_flatMap]
  have byStep : PoseidonTemplatePlan.plan.flatMap
      (fun step => step.sboxes.map (fun row => row.outputExpression)) =
      PoseidonTemplatePlan.plan.flatMap (fun step =>
        (PoseidonStepTrace.sboxProgram step.record.start step.record.step
          step.record.state).outputs) := by
    apply congrArg (fun projection => PoseidonTemplatePlan.plan.flatMap projection)
    funext step
    exact step.sboxPlan.outputExpressions_eq
  rw [byStep]
  change _ = PoseidonScheduleTrace.records.flatMap (fun record =>
    (PoseidonStepTrace.sboxProgram record.start record.step record.state).outputs)
  rw [← PoseidonTemplatePlan.plan_records_eq, List.flatMap_map]

/-- Satisfying source recipe rows establish every equation retained by the
compact S-box compiler. Input and retained-slot equalities are the adjacent
source projection contract, not independently supplied equation assumptions. -/
theorem equations_of_sourceRows
    {logicalWidth : Nat} (interface : Interface logicalWidth)
    (assignment : Assignment F logicalWidth) (env : Env)
    (one : assignment interface.oneColumn = 1)
    (inputEq : SparseLayer.evalState assignment interface.input =
      Layer.evalState env PoseidonScheduleTrace.canonicalState)
    (retained : ∀ row : Fin PoseidonRetainedSlots.rows.length,
      (interface.sboxOutput row).eval assignment =
        env (PoseidonRetainedSlots.rows.get row).step.output.val)
    (rows : ConstraintsHold env
      (recipeConstraints PoseidonScheduleTrace.inputCount
        (Permutation.compile PoseidonScheduleTrace.inputCount
          PoseidonScheduleTrace.canonicalState Permutation.schedule).recipes)) :
    SboxEquations interface assignment := by
  apply compile_equations interface assignment env one Permutation.schedule
    0 PoseidonScheduleTrace.inputCount interface.input
    PoseidonScheduleTrace.canonicalState inputEq _ rows
  intro index bound
  have rowBound : index < PoseidonRetainedSlots.rows.length := by
    rw [← retained_outputs_eq, List.length_map] at bound
    exact bound
  rw [Nat.zero_add, sboxOutputAt, dif_pos rowBound, retained]
  rw [← retained_outputs_eq]
  rw [List.getD_eq_getElem
    (l := PoseidonRetainedSlots.rows.map (fun row => row.outputExpression))
    (d := (0 : Expr)) (by simpa only [List.length_map] using rowBound),
    List.getElem_map]
  exact (PoseidonRetainedSlots.rows.get ⟨index, rowBound⟩).outputSound env

end NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness
