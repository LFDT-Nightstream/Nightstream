import NightstreamFPrime.Export.Stage1.NextPreimagePackage
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.R1CS.Support

/-!
Owns the five-row 14-matrix plan for HyperNova Construction 2's next-preimage
wiring. It reuses the retained prior/output preimage forms already owned by
PiCCS. No retained slot, source column, or private value is added.
-/

namespace NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def sourceRows : List R1CS.Row := NextPreimagePackage.sourceRows

theorem sourceRows_length : sourceRows.length = 5 :=
  NextPreimagePackage.sourceRows_length

theorem sourceRows_varsBelow :
    ∀ row ∈ sourceRows, row.VarsBelow Spartan.spartanColumnCount :=
  NextPreimagePackage.sourceRows_varsBelow

theorem sourceRows_varsSatisfy :
    ∀ row ∈ sourceRows,
      row.VarsSatisfy PiCCSOrdinarySourceSupport.Target := by
  rw [sourceRows, NextPreimagePackage.sourceRows_eq]
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy
    NextPreimagePackage.constraints NextPreimagePackage.privateStart
    PiCCSOrdinarySourceSupport.Target
    (NextPreimageInputs.spartanConstraints_varsSatisfy
      NextPreimagePackage.privateStart)
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro column support
  rcases support with source | fresh
  · exact source
  · rcases fresh with ⟨lower, upper⟩
    have noFresh : (NextPreimagePackage.lowered).next =
        NextPreimagePackage.privateStart := by
      rfl
    change column < NextPreimagePackage.lowered.next at upper
    rw [noFresh] at upper
    exact False.elim (by omega)

theorem sourceRows_rowCount_le :
    sourceRows.length ≤ 2 ^ Lifecycle.cubeVariables := by
  rw [sourceRows_length]
  norm_num [Lifecycle.cubeVariables]

def program : OrdinarySourcePlan.Program Spartan.spartanColumnCount where
  rowCount := sourceRows.length
  rowCount_le := sourceRows_rowCount_le
  row := fun index => sourceRows.get index
  bounded := fun index =>
    sourceRows_varsBelow _ (List.get_mem sourceRows index)

@[simp] theorem program_rowCount : program.rowCount = 5 := by
  exact sourceRows_length

theorem program_holds_iff_rowsHold (env : Env) :
    program.Holds env ↔ R1CS.RowsHold env sourceRows := by
  change (∀ index, (sourceRows.get index).Holds env) ↔
    R1CS.RowsHold env sourceRows
  constructor
  · intro holds row member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    exact holds index
  · intro holds index
    exact holds _ (List.get_mem _ index)

def inputs
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    program.Inputs logicalWidth where
  oneColumn := PiCCSOrdinaryRetainedGeometry.oneColumn geometry
  sourceMap := fun _ => PiCCSOrdinaryDirectPlan.sourceMap geometry

private theorem preservesCombination
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination)
    (scope : combination.VarsSatisfy PiCCSOrdinarySourceSupport.Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination
      (PiCCSOrdinaryDirectPlan.sourceMap geometry) assignment
      (RunningTransitionDirectPlan.transitionEnv application base)
      combination bounded := by
  intro term member
  exact PiCCSOrdinaryDirectPlan.sourceMap_form_eval_of_target geometry
    assignment base groupValue products encodes
    ⟨term.1, bounded term member⟩ (scope term member)

theorem inputs_preserve
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs geometry).sourceMap index) assignment
      (RunningTransitionDirectPlan.transitionEnv application base)
      (program.row index) (program.bounded index) := by
  intro index
  have scope := sourceRows_varsSatisfy _ (List.get_mem sourceRows index)
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

def plan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : ProductionRelation.Plan logicalWidth :=
  (program.compile (inputs geometry)).toPlan

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (plan geometry).rowCount = 5 := by
  rfl

theorem rowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1) :
    (plan geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        sourceRows := by
  rw [plan, OrdinarySourcePlan.Program.rowsZero_iff program (inputs geometry)
    assignment (RunningTransitionDirectPlan.transitionEnv application base)
    one (inputs_preserve geometry assignment base groupValue products encodes)]
  exact program_holds_iff_rowsHold _

theorem rowsZero_implies_spec
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (plan geometry).RowsZero assignment) :
    NextPreimage.SpecHolds NextPreimageInputs.sourceInterface
      RunningTransitionInputs.phaseOffset
      (Spartan.pullback
        (RunningTransitionDirectPlan.transitionEnv application base)) := by
  have sourceRowsHold := (rowsZero_iff_rowsHold geometry assignment base
    groupValue products encodes one).mp rows
  have spartanSpec := NextPreimagePackage.sourceRows_imply_spec
    (RunningTransitionDirectPlan.transitionEnv application base) sourceRowsHold
  have sourceSpec := (NextPreimageInputs.spartanSpec_iff_sourceSpec
    NextPreimagePackage.privateStart
    (RunningTransitionDirectPlan.transitionEnv application base)).mp spartanSpec
  refine {
    iteration := ?_
    initialState := fun index => ?_ }
  · simpa [NextPreimageInputs.sourceInterface] using sourceSpec.iteration
  · simpa [NextPreimageInputs.sourceInterface] using
      sourceSpec.initialState index

end NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan
