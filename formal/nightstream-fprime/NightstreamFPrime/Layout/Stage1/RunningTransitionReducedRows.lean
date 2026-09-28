import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation
import Mathlib.Tactic.LinearCombination

/-!
Owns the reduced quadratic rows and witness maps for the existing Stage 1
running-transition relation. One shared branch flag replaces the multiplication
scratch fields. The production package does not select this layout yet.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionReducedRows

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Layout NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionInputs RunningTransitionLayout

private abbrev LC := R1CS.LinearCombination

/-- Exact affine representation on the existing multiplication-free inputs. -/
private def linearForm : Expr → LC
  | .var index => R1CS.LinearCombination.ofVar index
  | .const value => R1CS.LinearCombination.const value
  | .add left right => R1CS.LinearCombination.add (linearForm left) (linearForm right)
  | .mul _ _ => R1CS.LinearCombination.zero

private theorem linearForm_eval (expression : Expr)
    (linear : R1CS.mulCount expression = 0) (env : Env) :
    (linearForm expression).eval env = expression.eval env := by
  induction expression with
  | var index => simp [linearForm]
  | const value => simp [linearForm]
  | add left right leftIH rightIH =>
      simp only [R1CS.mulCount, Nat.add_eq_zero_iff] at linear
      simp [linearForm, leftIH linear.1, rightIH linear.2]
  | mul left right => simp [R1CS.mulCount] at linear

def flagIndex : Nat := logicalColumnCount

private def flagForm : LC := R1CS.LinearCombination.ofVar flagIndex
private def baseForm : LC := R1CS.LinearCombination.add R1CS.LinearCombination.one (R1CS.LinearCombination.scale (-1) flagForm)
private def difference (left right : LC) : LC :=
  R1CS.LinearCombination.add left (R1CS.LinearCombination.scale (-1) right)

def flagRow : R1CS.Row :=
  ⟨linearForm iterationExpr, R1CS.LinearCombination.ofVar phaseOffset, flagForm⟩

def bindingRow : R1CS.Row :=
  ⟨linearForm iterationExpr, baseForm, R1CS.LinearCombination.zero⟩

def muxRow (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (index : RunningTransition.WordIndex) : R1CS.Row :=
  let default := R1CS.LinearCombination.const (RunningTransition.defaultWord
    (logicalWidth := logicalWidth) (publicFits := publicFits) index)
  ⟨flagForm,
    difference (linearForm (RunningTransition.runningWord
      (recursiveRunningExpr logicalWidth publicFits) index)) default,
    difference (linearForm (RunningTransition.runningWord
      (outputRunningExpr logicalWidth publicFits) index)) default⟩

def stateRow (index : RunningTransition.StateIndex) : R1CS.Row :=
  ⟨baseForm, difference (linearForm (initialStateExpr index))
    (linearForm (currentStateExpr index)), R1CS.LinearCombination.zero⟩

def rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  flagRow :: bindingRow ::
    (List.ofFn (muxRow logicalWidth publicFits) ++ List.ofFn stateRow)

private theorem flagRow_iff (env : Env) :
    flagRow.Holds env ↔
      env flagIndex = iterationExpr.eval env * env phaseOffset := by
  simp [flagRow, R1CS.Row.Holds, flagForm, linearForm, iterationExpr, eq_comm]

private theorem bindingRow_iff (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (env : Env)
    (flag : env flagIndex = iterationExpr.eval env * env phaseOffset) :
    bindingRow.Holds env ↔
      (RunningTransition.bindingConstraint
        (interface logicalWidth publicFits) phaseOffset).eval env = 0 := by
  simp [bindingRow, R1CS.Row.Holds, baseForm, flagForm, linearForm,
    RunningTransition.bindingConstraint, RunningTransition.baseFlag,
    RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
    interface, iterationExpr, ← sub_eq_add_neg, flag]

private theorem mux_algebra (flag recursive default output : ZMod goldilocksModulus) :
    flag * (recursive - default) = output - default ↔
      (1 - flag) * default + flag * recursive - output = 0 := by
  constructor <;> intro h <;> linear_combination h

/-- Exact source values read by each reduced running-word equation. -/
theorem muxRow_values {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (index : RunningTransition.WordIndex) :
    (muxRow logicalWidth publicFits index).a.eval env = env flagIndex ∧
      (muxRow logicalWidth publicFits index).b.eval env =
        (RunningTransition.runningWord (recursiveRunningExpr logicalWidth publicFits) index).eval env -
          RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits) index ∧
      (muxRow logicalWidth publicFits index).c.eval env =
        (RunningTransition.runningWord (outputRunningExpr logicalWidth publicFits) index).eval env -
          RunningTransition.defaultWord (logicalWidth := logicalWidth) (publicFits := publicFits) index := by
  have recursiveLinear := linearForm_eval _
    (runningWord_mulCount _ (recursiveMulFree relation) index) env
  have outputLinear := linearForm_eval _
    (runningWord_mulCount _ (outputMulFree logicalWidth publicFits) index) env
  simp only [muxRow, flagForm, difference, R1CS.LinearCombination.eval_ofVar,
    R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_scale,
    R1CS.LinearCombination.eval_const, recursiveLinear, outputLinear, neg_one_mul,
    ← sub_eq_add_neg, and_self]

private theorem muxRow_iff {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (flag : env flagIndex = iterationExpr.eval env * env phaseOffset)
    (index : RunningTransition.WordIndex) :
    (muxRow logicalWidth publicFits index).Holds env ↔
      (RunningTransition.muxConstraint
        (interface logicalWidth publicFits) phaseOffset index).eval env = 0 := by
  have recursiveLinear := linearForm_eval _
    (runningWord_mulCount _ (recursiveMulFree relation) index) env
  have outputLinear := linearForm_eval _
    (runningWord_mulCount _ (outputMulFree logicalWidth publicFits) index) env
  simp only [muxRow, R1CS.Row.Holds, flagForm, difference,
    R1CS.LinearCombination.eval_ofVar, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_const,
    recursiveLinear, outputLinear, neg_one_mul, ← sub_eq_add_neg,
    RunningTransition.muxConstraint, RunningTransition.baseFlag,
    RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
    Expr.eval_sub, Expr.eval_hmul, Expr.eval_hadd, Expr.eval_const,
    Expr.eval_var, interface]
  rw [flag]
  exact mux_algebra _ _ _ _

private theorem stateRow_iff (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (env : Env)
    (flag : env flagIndex = iterationExpr.eval env * env phaseOffset)
    (index : RunningTransition.StateIndex) :
    (stateRow index).Holds env ↔
      (RunningTransition.baseStateConstraint
        (interface logicalWidth publicFits) phaseOffset index).eval env = 0 := by
  simp [stateRow, R1CS.Row.Holds, baseForm, flagForm, difference, linearForm,
    RunningTransition.baseStateConstraint, RunningTransition.baseFlag,
    RunningTransition.recursiveFlag, RunningTransition.inverseExpr,
    interface, initialStateExpr, currentStateExpr, ← sub_eq_add_neg, flag]

/-- Equivalence to the existing logical constraints, with one bound flag. -/
theorem rows_iff_logical {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    R1CS.RowsHold env (rows logicalWidth publicFits) ↔
      env flagIndex = iterationExpr.eval env * env phaseOffset ∧
        ConstraintsHold env (logicalConstraints logicalWidth publicFits) := by
  constructor
  · intro holds
    have flag := (flagRow_iff env).mp (holds _ (by simp only [rows, List.mem_cons]; exact Or.inl trivial))
    refine ⟨flag, ?_⟩
    rw [logicalConstraints_eq]
    intro expression member
    simp only [RunningTransition.constraints, RunningTransition.muxConstraints,
      RunningTransition.baseStateConstraints, List.mem_cons,
      List.mem_append, List.mem_ofFn] at member
    rcases member with rfl | ⟨index, rfl⟩ | ⟨index, rfl⟩
    · apply (bindingRow_iff logicalWidth publicFits env flag).mp
      exact holds _ (by simp only [rows, List.mem_cons]; exact Or.inr (Or.inl trivial))
    · apply (muxRow_iff relation env flag index).mp
      exact holds _ (by simp only [rows, List.mem_cons, List.mem_append,
        List.mem_ofFn]; exact Or.inr (Or.inr (Or.inl ⟨index, rfl⟩)))
    · apply (stateRow_iff logicalWidth publicFits env flag index).mp
      exact holds _ (by simp only [rows, List.mem_cons, List.mem_append,
        List.mem_ofFn]; exact Or.inr (Or.inr (Or.inr ⟨index, rfl⟩)))
  · rintro ⟨flag, logical⟩ row member
    rw [logicalConstraints_eq] at logical
    simp only [rows, List.mem_cons, List.mem_append, List.mem_ofFn] at member
    rcases member with rfl | rfl | ⟨index, rfl⟩ | ⟨index, rfl⟩
    · exact (flagRow_iff env).mpr flag
    · apply (bindingRow_iff logicalWidth publicFits env flag).mpr
      exact logical _ (by simp only [RunningTransition.constraints, List.mem_cons]; exact Or.inl trivial)
    · apply (muxRow_iff relation env flag index).mpr
      exact logical _ (by
        simp only [RunningTransition.constraints, RunningTransition.muxConstraints,
          List.mem_cons, List.mem_append, List.mem_ofFn]
        exact Or.inr (Or.inl ⟨index, rfl⟩))
    · apply (stateRow_iff logicalWidth publicFits env flag index).mpr
      exact logical _ (by
        simp only [RunningTransition.constraints, RunningTransition.baseStateConstraints,
          List.mem_cons, List.mem_append, List.mem_ofFn]
        exact Or.inr (Or.inr ⟨index, rfl⟩))

theorem soundness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (holds : R1CS.RowsHold env (rows logicalWidth publicFits)) :
    RunningTransition.SpecHolds (interface logicalWidth publicFits)
      phaseOffset env := by
  apply RunningTransition.soundness _ env phaseOffset
    (assumptions logicalWidth publicFits relation env)
  apply holdsFlat_implies_holds
  exact ((rows_iff_logical relation env).mp holds).2

/-- The bound flag is a bit; it needs one low-norm coordinate. -/
theorem flag_boolean {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env) (holds : R1CS.RowsHold env (rows logicalWidth publicFits)) :
    env flagIndex = 0 ∨ env flagIndex = 1 := by
  have flag := (flagRow_iff env).mp (holds _ (by simp [rows]))
  have binding := holds bindingRow (by simp [rows])
  have product : iterationExpr.eval env * (1 - env flagIndex) = 0 := by
    simpa [bindingRow, R1CS.Row.Holds, baseForm, flagForm, linearForm,
      iterationExpr, ← sub_eq_add_neg] using binding
  rcases GoldilocksPrime.baseFieldNoZeroDivisors _ _ product with zero | one
  · exact Or.inl (by rw [flag, zero, zero_mul])
  · exact Or.inr (sub_eq_zero.mp one).symm

/-- Executable witness extension. Only the new shared flag is written. -/
def extend (env : Env) : Env :=
  Env.set env flagIndex (iterationExpr.eval env * env phaseOffset)

theorem extend_agrees (env : Env) : AgreesOutside env (extend env) flagIndex 1 := by
  intro index outside
  exact Env.set_of_ne env flagIndex index _ (by omega)


private theorem logical_scope {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    ∀ expression ∈ logicalConstraints logicalWidth publicFits,
      expression.VarsBelow flagIndex := by
  have scope := RunningTransition.flatConstraints_varsBelow
    (interface logicalWidth publicFits) phaseOffset env
    (assumptions logicalWidth publicFits relation env)

  rw [RunningTransition.flatConstraints_operations] at scope
  rw [logicalConstraints_eq]
  exact scope

private theorem extend_below (env : Env) (index : Nat) (below : index < flagIndex) :
    extend env index = env index :=
  extend_agrees env index (Or.inl below)

private theorem extended_flag {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    extend env flagIndex = iterationExpr.eval (extend env) * extend env phaseOffset := by
  have phaseBelow : phaseOffset < flagIndex := by
    change phaseOffset < phaseOffset + 1
    omega
  have iterationBelow : iterationExpr.VarsBelow flagIndex := by
    have original := (assumptions logicalWidth publicFits relation env).iteration
    exact Expr.VarsBelow.mono iterationExpr original (by
      change phaseOffset ≤ phaseOffset + 1
      omega)
  rw [Expr.eval_eq_of_agree_below _ flagIndex (extend env) env iterationBelow
    (extend_below env), extend_below env phaseOffset phaseBelow]
  exact Env.set_self env flagIndex _

/-- Construct the reduced witness from any satisfying original logical witness. -/
theorem extend_complete {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (logical : ConstraintsHold env
      (logicalConstraints logicalWidth publicFits)) :
    R1CS.RowsHold (extend env) (rows logicalWidth publicFits) := by
  apply (rows_iff_logical relation (extend env)).mpr
  refine ⟨extended_flag relation env, ?_⟩
  exact constraintsHold_of_agree_below env (extend env)
    (logicalConstraints logicalWidth publicFits) flagIndex
    (logical_scope relation env) (extend_below env) logical

/-- Constructive completeness preserves every input and changes two local cells:
the existing inverse field and the new shared flag. -/
theorem completeness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env) :
    ∃ completed, AgreesOutside env completed phaseOffset 2 ∧
      R1CS.RowsHold completed (rows logicalWidth publicFits) := by
  rcases RunningTransition.completeness (interface logicalWidth publicFits) env phaseOffset
      (assumptions logicalWidth publicFits relation env) specification with
    ⟨logical, agrees, holds⟩
  have oldLogical : ConstraintsHold logical (logicalConstraints logicalWidth publicFits) := by
    rw [logicalConstraints_eq]
    unfold holdsFlat at holds
    rw [RunningTransition.flatConstraints_operations] at holds
    exact holds
  refine ⟨extend logical, ?_, extend_complete relation logical oldLogical⟩
  have inverseAgrees : AgreesOutside env logical phaseOffset 1 := by
    simpa only [RunningTransition.localLength_eq, RunningTransition.exactPrivateCount]
      using agrees
  exact inverseAgrees.append (extend_agrees logical)

private theorem reconstruct_plan (plan : R1CS.LoweringPlan) (env : Env)
    (scope : ∀ expression ∈ plan.constraints, expression.VarsBelow plan.firstFresh)
    (logical : ConstraintsHold env plan.constraints) :
    R1CS.RowsHold (R1CS.executeConstraints env plan.constraints plan.firstFresh) plan.rows ∧
      AgreesOutside env (R1CS.executeConstraints env plan.constraints plan.firstFresh)
        plan.firstFresh plan.freshColumnCount := by
  exact ⟨R1CS.executeConstraints_holds_rows env plan.constraints plan.firstFresh scope logical,
    R1CS.executeConstraints_agreesOutside env plan.constraints plan.firstFresh⟩

/-- Executable recovery of every old R1CS scratch value from a reduced witness. -/
def reconstruct (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (env : Env) : Env :=
  R1CS.executeConstraints env (plan logicalWidth publicFits).constraints
    (plan logicalWidth publicFits).firstFresh

theorem reconstruct_complete {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (holds : R1CS.RowsHold env (rows logicalWidth publicFits)) :
    PhysicalHolds logicalWidth publicFits (reconstruct logicalWidth publicFits env) ∧
      AgreesOutside env (reconstruct logicalWidth publicFits env) flagIndex 296137 := by
  have scope : ∀ expression ∈ (plan logicalWidth publicFits).constraints,
      expression.VarsBelow (plan logicalWidth publicFits).firstFresh := by
    rw [plan_constraints, plan_firstFresh]
    exact logical_scope relation env
  have logical : ConstraintsHold env (plan logicalWidth publicFits).constraints := by
    rw [plan_constraints]
    exact ((rows_iff_logical relation env).mp holds).2
  have result := reconstruct_plan (plan logicalWidth publicFits) env scope logical
  refine ⟨?_, ?_⟩
  · rw [PhysicalHolds, physicalRows, reconstruct]
    exact result.1
  · have agrees := result.2
    rw [plan_firstFresh] at agrees
    have count : (plan logicalWidth publicFits).freshColumnCount = 296137 :=
      physicalFreshColumnCount_eq relation
    rw [count] at agrees
    exact agrees

/-- The original physical witness maps back without changing any old logical input. -/
theorem original_to_reduced {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (holds : PhysicalHolds logicalWidth publicFits env) :
    R1CS.RowsHold (extend env) (rows logicalWidth publicFits) := by
  apply extend_complete relation env
  have logical := R1CS.LoweringPlan.sound (plan logicalWidth publicFits) env holds
  rw [plan_constraints] at logical
  exact logical

private def firstScratchRow : R1CS.Row :=
  ⟨R1CS.LinearCombination.ofVar
      (PilotProduction.priorPreimageStart + RunningTransitionInputs.iterationWordIndex),
    R1CS.LinearCombination.ofVar RunningTransitionInputs.phaseOffset,
    R1CS.LinearCombination.ofVar flagIndex⟩

private theorem lowerHead_member (expression : Expr) (rest : List Expr) (start : Nat)
    (row : R1CS.Row) (member : row ∈ (R1CS.lowerConstraint expression start).rows) :
    row ∈ (R1CS.lowerConstraints (expression :: rest) start).rows :=
  List.mem_append_left _ member

/-- The old lowering already stores the required flag at its first scratch cell. -/
theorem physical_firstScratch {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env) (physical : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits env) :
    env flagIndex = RunningTransitionInputs.iterationExpr.eval env *
      env RunningTransitionInputs.phaseOffset := by
  let binding := Stage1.RunningTransition.bindingConstraint
    (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset
  have member : firstScratchRow ∈ (R1CS.lowerConstraint binding flagIndex).rows := by
    change firstScratchRow ∈ firstScratchRow :: _
    exact List.mem_cons_self
  have holds : firstScratchRow.Holds env := by
    apply physical firstScratchRow
    rw [RunningTransitionLayout.physicalRows, R1CS.LoweringPlan.rows,
      R1CS.LoweringPlan.lowering, RunningTransitionLayout.plan_constraints,
      RunningTransitionLayout.plan_firstFresh, RunningTransitionLayout.logicalConstraints_eq,
      Stage1.RunningTransition.constraints]
    exact lowerHead_member binding
      (Stage1.RunningTransition.muxConstraints
          (RunningTransitionInputs.interface logicalWidth publicFits)
          RunningTransitionInputs.phaseOffset ++
        Stage1.RunningTransition.baseStateConstraints
          (RunningTransitionInputs.interface logicalWidth publicFits)
          RunningTransitionInputs.phaseOffset)
      flagIndex firstScratchRow member
  symm
  simpa [firstScratchRow, R1CS.Row.Holds, RunningTransitionInputs.iterationExpr] using holds

/-- The existing full physical witness already satisfies the reduced rows as is. -/
theorem physical_already_reduced {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (physical : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits env) :
    R1CS.RowsHold env (rows logicalWidth publicFits) := by
  have reduced := original_to_reduced relation env physical
  have same : extend env = env := by
    funext column
    simp only [extend, Env.set]
    split_ifs with same
    · subst column
      exact (physical_firstScratch env physical).symm
    · rfl
  rw [same] at reduced
  exact reduced

/-- The existing first scratch value is admissible as one low-norm bit coordinate. -/
theorem physical_firstScratch_boolean {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) (physical : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits env) :
    env flagIndex = 0 ∨ env flagIndex = 1 :=
  flag_boolean env (physical_already_reduced relation env physical)

theorem flag_low_norm {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env) (holds : R1CS.RowsHold env (rows logicalWidth publicFits)) :
    centeredMagnitude
      (env flagIndex) < 2 := by
  rcases flag_boolean env holds with zero | one
  · rw [zero]; decide
  · rw [one]; decide

theorem row_count (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (rows logicalWidth publicFits).length = 49359 := by
  simp only [rows, List.length_cons, List.length_append, List.length_ofFn]
  rfl

theorem local_geometry :
    296137 * 41 - 1 = 12141616 ∧ 345495 - 49359 = 296136 := by decide

end NightstreamFPrime.Layout.Stage1.RunningTransitionReducedRows
