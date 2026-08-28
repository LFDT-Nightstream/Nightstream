import NightstreamFPrime.Layout.ProductionRelation.PoseidonScheduleTrace
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSourceRows

/-!
Owns the fully compiled, constant-size direct-row plan for one canonical
Poseidon2 permutation template. Every source expression is checked by the
fail-closed bounded affine or exact-variable recognizer.

The plan is shared by every package invocation. This module does not expand
the invocation list.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonTemplatePlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Record := PoseidonScheduleTrace.Record

/-- A fail-closed compiled S-box list, with exact source-list custody. -/
structure SboxListPlan (inputs outputs : List Expr) where
  rows : List
    (PoseidonSourceRows.SboxSource PoseidonScheduleTrace.sourceColumnCount)
  inputExpressions_eq :
    rows.map (fun row => row.inputExpression) = inputs
  outputExpressions_eq :
    rows.map (fun row => row.outputExpression) = outputs

private theorem compileSbox?_expressions {input output : Expr}
    {row : PoseidonSourceRows.SboxSource
      PoseidonScheduleTrace.sourceColumnCount}
    (found : PoseidonSourceRows.compileSbox?
      PoseidonScheduleTrace.sourceColumnCount input output = some row) :
    row.inputExpression = input ∧ row.outputExpression = output := by
  unfold PoseidonSourceRows.compileSbox? at found
  split at found <;> try contradiction
  split at found <;> try contradiction
  have row_eq := Option.some.inj found
  subst row
  exact ⟨rfl, rfl⟩

/-- Compile two equal-length S-box expression lists and retain exact custody
of both lists in the result type. -/
def compileSboxList? :
    (inputs outputs : List Expr) → Option (SboxListPlan inputs outputs)
  | [], [] =>
      some
        { rows := []
          inputExpressions_eq := rfl
          outputExpressions_eq := rfl }
  | input :: inputs, output :: outputs =>
      match found : PoseidonSourceRows.compileSbox?
          PoseidonScheduleTrace.sourceColumnCount input output with
      | none => none
      | some row =>
          match compileSboxList? inputs outputs with
          | none => none
          | some tail =>
              some
                { rows := row :: tail.rows
                  inputExpressions_eq := by
                    rw [List.map_cons, (compileSbox?_expressions found).1,
                      tail.inputExpressions_eq]
                  outputExpressions_eq := by
                    rw [List.map_cons, (compileSbox?_expressions found).2,
                      tail.outputExpressions_eq] }
  | _, _ => none

/-- A fail-closed compiled output list, with exact source-list custody. -/
structure OutputListPlan (outputs linears : List Expr) where
  rows : List
    (PoseidonSourceRows.OutputSource PoseidonScheduleTrace.sourceColumnCount)
  outputExpressions_eq :
    rows.map (fun row => row.outputExpression) = outputs
  linearExpressions_eq :
    rows.map (fun row => row.linearExpression) = linears

private theorem compileOutput?_expressions {output linear : Expr}
    {row : PoseidonSourceRows.OutputSource
      PoseidonScheduleTrace.sourceColumnCount}
    (found : PoseidonSourceRows.compileOutput?
      PoseidonScheduleTrace.sourceColumnCount output linear = some row) :
    row.outputExpression = output ∧ row.linearExpression = linear := by
  unfold PoseidonSourceRows.compileOutput? at found
  split at found <;> try contradiction
  split at found <;> try contradiction
  have row_eq := Option.some.inj found
  subst row
  exact ⟨rfl, rfl⟩

/-- Compile two equal-length output expression lists and retain exact custody
of both lists in the result type. -/
def compileOutputList? :
    (outputs linears : List Expr) → Option (OutputListPlan outputs linears)
  | [], [] =>
      some
        { rows := []
          outputExpressions_eq := rfl
          linearExpressions_eq := rfl }
  | output :: outputs, linear :: linears =>
      match found : PoseidonSourceRows.compileOutput?
          PoseidonScheduleTrace.sourceColumnCount output linear with
      | none => none
      | some row =>
          match compileOutputList? outputs linears with
          | none => none
          | some tail =>
              some
                { rows := row :: tail.rows
                  outputExpressions_eq := by
                    rw [List.map_cons, (compileOutput?_expressions found).1,
                      tail.outputExpressions_eq]
                  linearExpressions_eq := by
                    rw [List.map_cons, (compileOutput?_expressions found).2,
                      tail.linearExpressions_eq] }
  | _, _ => none

structure StepPlan where
  record : Record
  sboxPlan : SboxListPlan
    (PoseidonStepTrace.sboxInputs record.step record.state)
    (PoseidonStepTrace.sboxProgram record.start record.step record.state).outputs
  outputPlan : OutputListPlan
    (List.ofFn (Permutation.stepOutput record.start record.step))
    (List.ofFn
      (PoseidonStepTrace.outputExpressions record.start record.step record.state))

def StepPlan.sboxes (step : StepPlan) := step.sboxPlan.rows

def StepPlan.outputs (step : StepPlan) := step.outputPlan.rows

def compileRecord? (record : Record) : Option StepPlan := do
  let sboxPlan ← compileSboxList?
    (PoseidonStepTrace.sboxInputs record.step record.state)
    (PoseidonStepTrace.sboxProgram record.start record.step record.state).outputs
  let outputPlan ← compileOutputList?
    (List.ofFn (Permutation.stepOutput record.start record.step))
    (List.ofFn
      (PoseidonStepTrace.outputExpressions record.start record.step record.state))
  pure { record := record, sboxPlan := sboxPlan, outputPlan := outputPlan }

def compilePlan? : Option (List StepPlan) :=
  PoseidonScheduleTrace.records.mapM compileRecord?

/-- Executable production plan. The success theorem below proves the default
branch is unreachable. -/
def plan : List StepPlan := compilePlan?.getD []

/-- Every one of the fixed template's 334 expression pairs passes exact
bounded compilation. -/
theorem compilePlan?_eq : compilePlan? = some plan := by
  rfl

@[simp] theorem plan_length : plan.length = 31 := by
  rfl

/-- The compiled plan covers the canonical schedule ledger in exact order. -/
@[simp] theorem plan_records_eq :
    plan.map (fun step => step.record) = PoseidonScheduleTrace.records := by
  rfl

@[simp] theorem sboxRowCount_eq :
    (plan.map fun step => step.sboxes.length).sum = 86 := by
  rfl

@[simp] theorem outputRowCount_eq :
    (plan.map fun step => step.outputs.length).sum = 248 := by
  rfl

@[simp] theorem directRowCount_eq :
    (plan.map fun step => step.sboxes.length + step.outputs.length).sum = 334 := by
  rfl

private theorem map_eq_map_of_forall_mem {α β : Type}
    (rows : List α) (left right : α → β)
    (equal : ∀ row ∈ rows, left row = right row) :
    rows.map left = rows.map right := by
  induction rows with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.map_cons]
      rw [equal head (by simp)]
      exact congrArg (List.cons (right head))
        (induction fun row member => equal row (by simp [member]))

/-- Exact direct-row equations required by one canonical schedule record. -/
def StepPlan.Holds (step : StepPlan) (env : Env) : Prop :=
  (∀ row ∈ step.sboxes,
      row.outputExpression.eval env =
        Layer.sboxF (row.inputExpression.eval env)) ∧
    ∀ row ∈ step.outputs,
      row.outputExpression.eval env = row.linearExpression.eval env

/-- Every compiled direct row of one schedule step has zero selective
residual on the final assignment. -/
def StepPlan.RowsZero {logicalWidth : Nat} (step : StepPlan)
    (sourceMap : SourceCompiler.SourceMap
      PoseidonScheduleTrace.sourceColumnCount logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth) :
    Prop :=
  (∀ row ∈ step.sboxes,
      (row.step.compile sourceMap oneColumn).residual assignment = 0) ∧
    ∀ row ∈ step.outputs,
      (row.step.compile sourceMap oneColumn).residual assignment = 0

/-- Every direct row of every compiled schedule step has zero residual. -/
def PlanRowsZero {logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap
      PoseidonScheduleTrace.sourceColumnCount logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth) :
    Prop :=
  ∀ step ∈ plan, step.RowsZero sourceMap oneColumn assignment

private theorem seventhPower_eq_sboxF (value : F) :
    Spec.ProductionRelation.RowSemantics.seventhPower value =
      Layer.sboxF value := by
  simp [Spec.ProductionRelation.RowSemantics.seventhPower,
    CCSResidualTable.pow, ConcreteCarrier.baseOps, Layer.sboxF,
    Spec.Poseidon2.sbox, mul_assoc]

/-- Zero residuals of the compiled sparse rows are equivalent to the exact
retained trace equations under the proved source substitution. -/
theorem StepPlan.rowsZero_iff_holds {logicalWidth : Nat} (step : StepPlan)
    (sourceMap : SourceCompiler.SourceMap
      PoseidonScheduleTrace.sourceColumnCount logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (env : Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment env) :
    step.RowsZero sourceMap oneColumn assignment ↔ step.Holds env := by
  constructor
  · rintro ⟨sboxes, outputs⟩
    constructor
    · intro row member
      have zero := (PoseidonSourceRows.SboxSource.residual_zero_iff row
        sourceMap oneColumn assignment env one preserves).mp
          (sboxes row member)
      rw [← seventhPower_eq_sboxF]
      exact zero.symm
    · intro row member
      exact (PoseidonSourceRows.OutputSource.residual_zero_iff row
        sourceMap oneColumn assignment env one preserves).mp
          (outputs row member)
  · rintro ⟨sboxes, outputs⟩
    constructor
    · intro row member
      apply (PoseidonSourceRows.SboxSource.residual_zero_iff row
        sourceMap oneColumn assignment env one preserves).mpr
      rw [seventhPower_eq_sboxF]
      exact (sboxes row member).symm
    · intro row member
      exact (PoseidonSourceRows.OutputSource.residual_zero_iff row
        sourceMap oneColumn assignment env one preserves).mpr
          (outputs row member)

/-- The compiled expression lists are exactly the retained mathematical trace
of their canonical Poseidon2 schedule record. -/
theorem StepPlan.holds_implies_trace (step : StepPlan) (env : Env)
    (holds : step.Holds env) :
    PoseidonStepTrace.Holds env step.record.start step.record.step
      step.record.state := by
  rcases holds with ⟨sboxes, outputs⟩
  constructor
  · have outputExpressions := congrArg (List.map (Expr.eval env))
      step.sboxPlan.outputExpressions_eq
    have inputExpressions :=
      congrArg (List.map fun input => Layer.sboxF (input.eval env))
        step.sboxPlan.inputExpressions_eq
    calc
      (PoseidonStepTrace.sboxProgram step.record.start step.record.step
          step.record.state).outputs.map (Expr.eval env) =
          step.sboxes.map (fun row => row.outputExpression.eval env) := by
        simpa [StepPlan.sboxes, List.map_map] using outputExpressions.symm
      _ = step.sboxes.map (fun row =>
          Layer.sboxF (row.inputExpression.eval env)) :=
        map_eq_map_of_forall_mem step.sboxes _ _ sboxes
      _ = (PoseidonStepTrace.sboxInputs step.record.step
          step.record.state).map
            (fun input => Layer.sboxF (input.eval env)) := by
        simpa [StepPlan.sboxes, List.map_map] using inputExpressions
  · apply List.ofFn_injective
    change
      List.ofFn (fun lane =>
          (Permutation.stepOutput step.record.start step.record.step lane).eval env) =
        List.ofFn (fun lane =>
          (PoseidonStepTrace.outputExpressions step.record.start
              step.record.step step.record.state lane).eval env)
    have outputExpressions := congrArg (List.map (Expr.eval env))
      step.outputPlan.outputExpressions_eq
    have linearExpressions := congrArg (List.map (Expr.eval env))
      step.outputPlan.linearExpressions_eq
    calc
      List.ofFn (fun lane =>
          (Permutation.stepOutput step.record.start step.record.step lane).eval env) =
          (List.ofFn (Permutation.stepOutput step.record.start
            step.record.step)).map (Expr.eval env) :=
        List.ofFn_comp' _ _
      _ = step.outputs.map (fun row => row.outputExpression.eval env) := by
        simpa [StepPlan.outputs, List.map_map] using outputExpressions.symm
      _ = step.outputs.map (fun row => row.linearExpression.eval env) :=
        map_eq_map_of_forall_mem step.outputs _ _ outputs
      _ = (List.ofFn (PoseidonStepTrace.outputExpressions step.record.start
          step.record.step step.record.state)).map (Expr.eval env) := by
        simpa [StepPlan.outputs, List.map_map] using linearExpressions
      _ = List.ofFn (fun lane =>
          (PoseidonStepTrace.outputExpressions step.record.start
            step.record.step step.record.state lane).eval env) :=
        (List.ofFn_comp' _ _).symm

/-- Every satisfying compiled step has the exact mathematical Poseidon2 step
semantics. -/
theorem StepPlan.holds_implies_step_sound (step : StepPlan) (env : Env)
    (holds : step.Holds env) :
    Layer.evalState env
        (Permutation.stepOutput step.record.start step.record.step) =
      Permutation.applyF step.record.step
        (Layer.evalState env step.record.state) :=
  PoseidonStepTrace.holds_implies_step_sound env step.record.start
    step.record.step step.record.state (step.holds_implies_trace env holds)

/-- The final sparse selective rows imply the exact mathematical Poseidon2
step, through the proved source substitution. -/
theorem StepPlan.rowsZero_implies_step_sound {logicalWidth : Nat}
    (step : StepPlan)
    (sourceMap : SourceCompiler.SourceMap
      PoseidonScheduleTrace.sourceColumnCount logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (env : Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment env)
    (rows : step.RowsZero sourceMap oneColumn assignment) :
    Layer.evalState env
        (Permutation.stepOutput step.record.start step.record.step) =
      Permutation.applyF step.record.step
        (Layer.evalState env step.record.state) :=
  step.holds_implies_step_sound env
    ((step.rowsZero_iff_holds sourceMap oneColumn assignment env one
      preserves).mp rows)

/-- Zero residuals of the complete compiled template imply the exact
reference Poseidon2 permutation in the final retained output lanes. -/
theorem planRowsZero_implies_permute {logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap
      PoseidonScheduleTrace.sourceColumnCount logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (env : Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment env)
    (rows : PlanRowsZero sourceMap oneColumn assignment) :
    List.ofFn (Layer.evalState env
        (Permutation.scheduleOutput PoseidonScheduleTrace.inputCount)) =
      Spec.Poseidon2.permute
        (List.ofFn (Layer.evalState env
          PoseidonScheduleTrace.canonicalState)) := by
  apply PoseidonScheduleTrace.records_imply_permute
  intro record recordMember
  have planMember : record ∈ plan.map (fun step => step.record) := by
    rw [plan_records_eq]
    exact recordMember
  rcases List.mem_map.mp planMember with ⟨step, stepMember, stepRecord⟩
  subst record
  exact step.rowsZero_implies_step_sound sourceMap oneColumn assignment env one
    preserves (rows step stepMember)

end NightstreamFPrime.Layout.ProductionRelation.PoseidonTemplatePlan
