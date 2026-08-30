import NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan

/-!
Owns the executable source resolver and direct 14-matrix plan for one
verifier-selected Stage 1 application.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ApplicationRetainedBlocks
open ApplicationRetainedGeometry

def InRange (start count column : Nat) : Prop :=
  start ≤ column ∧ column < start + count

local instance (start count column : Nat) : Decidable (InRange start count column) := by
  unfold InRange
  infer_instance

def rangeIndex {start count column : Nat}
    (inside : InRange start count column) : Fin count :=
  ⟨column - start, by unfold InRange at inside; omega⟩

@[simp] theorem rangeIndex_source {start count column : Nat}
    (inside : InRange start count column) :
    start + (rangeIndex inside).val = column := by
  change start + (column - start) = column
  unfold InRange at inside
  omega

inductive Location (application : Lifecycle.Stage1.Application.Program) where
  | input (index : Lifecycle.Stage1.Application.StateIndex)
  | witness (index : Fin application.witnessWordCount)
  | output (index : Lifecycle.Stage1.Application.StateIndex)
  | localValues (index : Fin (localCount application))

namespace Location

def sourceColumn {application : Lifecycle.Stage1.Application.Program} :
    Location application → Nat
  | .input index => Layout.Stage1.ApplicationInputs.inputColumn index
  | .witness index => Layout.Stage1.ApplicationInputs.witnessColumn index
  | .output index => Layout.Stage1.ApplicationInputs.outputColumn index
  | .localValues index =>
      Layout.Stage1.ApplicationInputs.localStart application + index.val

def form {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    Location application → SparseForm logicalWidth
  | .input index => (inputBlock application).form
      (inputStart application) (inputFits geometry) index
  | .witness index => (witnessBlock application).form
      (witnessStart application) (witnessFits geometry) index
  | .output index => (outputBlock application).form
      (outputStart application) (outputFits geometry) index
  | .localValues index => (localBlock application).form
      (localStart application) (localFits geometry) index

theorem form_eval {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth application) → F)
    (encodes : Encodes geometry assignment source)
    (location : Location application) :
    (location.form geometry).eval assignment =
      source ⟨location.sourceColumn, by
        cases location with
        | input index => exact (inputBlock application).source index |>.isLt
        | witness index => exact (witnessBlock application).source index |>.isLt
        | output index => exact (outputBlock application).source index |>.isLt
        | localValues index => exact (localBlock application).source index |>.isLt⟩ := by
  cases location with
  | input index =>
      exact LowNormBlock.Block.form_eval _ _ _ assignment source encodes.input
        index
  | witness index =>
      exact LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.witness index
  | output index =>
      exact LowNormBlock.Block.form_eval _ _ _ assignment source encodes.output
        index
  | localValues index =>
      exact LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.localValues index

end Location

structure Located (application : Lifecycle.Stage1.Application.Program)
    (column : Nat) where
  location : Location application
  owns : location.sourceColumn = column

def classifySource (application : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Option (Located application column) :=
  if input : InRange Layout.Stage1.ApplicationInputs.currentWordStart
      Lifecycle.Stage1.Application.stateWordCount column then
    some ⟨.input (rangeIndex input), by
      rw [Location.sourceColumn,
        Layout.Stage1.ApplicationInputs.inputColumn_value,
        rangeIndex_source input]⟩
  else if witness : InRange Layout.Stage1.ApplicationInputs.witnessStart
      application.witnessWordCount column then
    some ⟨.witness (rangeIndex witness), by
      rw [Location.sourceColumn]
      unfold Layout.Stage1.ApplicationInputs.witnessColumn
      exact rangeIndex_source witness⟩
  else if output : InRange 45972
      Lifecycle.Stage1.Application.stateWordCount column then
    some ⟨.output (rangeIndex output), by
      rw [Location.sourceColumn,
        Layout.Stage1.ApplicationInputs.outputColumn_value,
        rangeIndex_source output]⟩
  else if localValues : InRange
      (Layout.Stage1.ApplicationInputs.localStart application)
      (localCount application) column then
    some ⟨.localValues (rangeIndex localValues), by
      rw [Location.sourceColumn, rangeIndex_source localValues]⟩
  else
    none

theorem classifySource_complete
    (application : Lifecycle.Stage1.Application.Program) {column : Nat}
    (support : ApplicationDirectSource.SourceAllowed application column) :
    (classifySource application column).isSome := by
  rcases support with input | witness | output | localSupport
  · rcases input with ⟨index, rfl⟩
    have inside : InRange Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
        (Layout.Stage1.ApplicationInputs.inputColumn index) := by
      rw [Layout.Stage1.ApplicationInputs.inputColumn_value]
      exact ⟨by omega, by have := index.isLt; omega⟩
    unfold classifySource
    rw [dif_pos inside]
    rfl
  · rcases witness with ⟨index, rfl⟩
    have notInput : ¬ InRange Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
        (Layout.Stage1.ApplicationInputs.witnessColumn index) := by
      unfold InRange Layout.Stage1.ApplicationInputs.witnessColumn
        Layout.Stage1.ApplicationInputs.witnessStart
        Layout.Stage1.Spartan.privateColumnCount
        Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
      omega
    have inside : InRange Layout.Stage1.ApplicationInputs.witnessStart
        application.witnessWordCount
        (Layout.Stage1.ApplicationInputs.witnessColumn index) := by
      unfold InRange Layout.Stage1.ApplicationInputs.witnessColumn
      exact ⟨by omega, by have := index.isLt; omega⟩
    unfold classifySource
    rw [dif_neg notInput, dif_pos inside]
    rfl
  · rcases output with ⟨index, rfl⟩
    have notInput : ¬ InRange Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
        (Layout.Stage1.ApplicationInputs.outputColumn index) := by
      rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
      unfold InRange Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
      omega
    have notWitness : ¬ InRange Layout.Stage1.ApplicationInputs.witnessStart
        application.witnessWordCount
        (Layout.Stage1.ApplicationInputs.outputColumn index) := by
      rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
      unfold InRange
      have indexBound := index.isLt
      norm_num [Layout.Stage1.ApplicationInputs.witnessStart,
        Layout.Stage1.Spartan.privateColumnCount,
        Lifecycle.Stage1.Application.stateWordCount] at indexBound ⊢
      omega
    have inside : InRange 45972 Lifecycle.Stage1.Application.stateWordCount
        (Layout.Stage1.ApplicationInputs.outputColumn index) := by
      rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
      exact ⟨by omega, by have := index.isLt; omega⟩
    unfold classifySource
    rw [dif_neg notInput, dif_neg notWitness, dif_pos inside]
    rfl
  · have notInput : ¬ InRange
        Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount column := by
      unfold InRange Layout.Stage1.ApplicationInputs.currentWordStart
        Lifecycle.Stage1.Application.stateWordCount
      have startLarge : 39 ≤
          Layout.Stage1.ApplicationInputs.localStart application := by
        unfold Layout.Stage1.ApplicationInputs.localStart
          Layout.Stage1.ApplicationInputs.witnessStart
          Layout.Stage1.Spartan.privateColumnCount
        omega
      omega
    have notWitness : ¬ InRange Layout.Stage1.ApplicationInputs.witnessStart
        application.witnessWordCount column := by
      unfold InRange
      unfold Layout.Stage1.ApplicationInputs.localStart at localSupport
      omega
    have notOutput : ¬ InRange 45972
        Lifecycle.Stage1.Application.stateWordCount column := by
      unfold InRange Lifecycle.Stage1.Application.stateWordCount
      have startLarge : 45976 ≤
          Layout.Stage1.ApplicationInputs.localStart application := by
        unfold Layout.Stage1.ApplicationInputs.localStart
          Layout.Stage1.ApplicationInputs.witnessStart
          Layout.Stage1.Spartan.privateColumnCount
        omega
      omega
    have inside : InRange (Layout.Stage1.ApplicationInputs.localStart application)
        (localCount application) column := by
      unfold InRange localCount
      have startLe := ApplicationRetainedBlocks.localStart_le_sourceWidth
        application
      unfold ApplicationRetainedBlocks.sourceWidth at startLe ⊢
      omega
    unfold classifySource
    rw [dif_neg notInput, dif_neg notWitness, dif_neg notOutput,
      dif_pos inside]
    rfl

def sourceMap {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth) :
    SourceCompiler.SourceMap (sourceWidth application) logicalWidth where
  form := fun column =>
    match classifySource application column.val with
    | none => .empty
    | some located => located.location.form geometry

def sourceEnv {application : Lifecycle.Stage1.Application.Program}
    (source : Fin (sourceWidth application) → F) : Env :=
  fun column => if bound : column < sourceWidth application then
    source ⟨column, bound⟩ else 0

theorem sourceMap_form_eval
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth application) → F)
    (encodes : Encodes geometry assignment source)
    (column : Fin (sourceWidth application))
    (support : ApplicationDirectSource.SourceAllowed application column.val) :
    ((sourceMap geometry).form column).eval assignment =
      sourceEnv source column.val := by
  have complete := classifySource_complete application support
  cases found : classifySource application column.val with
  | none => simp [found] at complete
  | some located =>
      change (match classifySource application column.val with
        | none => SparseForm.empty
        | some value => value.location.form geometry).eval assignment = _
      rw [found, Location.form_eval geometry assignment source encodes]
      unfold sourceEnv
      rw [dif_pos column.isLt]
      apply congrArg source
      apply Fin.ext
      exact located.owns

private theorem preservesCombination
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth application) → F)
    (encodes : Encodes geometry assignment source)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded (sourceWidth application)
      combination)
    (scope : combination.VarsSatisfy
      (ApplicationDirectSource.SourceAllowed application)) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (sourceEnv source) combination bounded := by
  intro term member
  exact sourceMap_form_eval geometry assignment source encodes
    ⟨term.1, bounded term member⟩ (scope term member)

private theorem programRow_support
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount) :
    ((ApplicationDirectSource.program application fits).row index).VarsSatisfy
      (ApplicationDirectSource.SourceAllowed application) := by
  exact ApplicationDirectSource.sourceRows_varsSatisfy application _
    (List.get_mem _ index)

def inputs
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth) :
    (ApplicationDirectSource.program application fits).Inputs logicalWidth where
  oneColumn := oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth application) → F)
    (encodes : Encodes geometry assignment source) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs fits geometry).sourceMap index) assignment (sourceEnv source)
      ((ApplicationDirectSource.program application fits).row index)
      ((ApplicationDirectSource.program application fits).bounded index) := by
  intro index
  have scope := programRow_support application fits index
  exact ⟨
    preservesCombination geometry assignment source encodes _ _ scope.1,
    preservesCombination geometry assignment source encodes _ _ scope.2.1,
    preservesCombination geometry assignment source encodes _ _ scope.2.2⟩

def plan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ((ApplicationDirectSource.program application fits).compile
    (inputs fits geometry)).toPlan

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth) :
    (plan fits geometry).rowCount =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  change ((ApplicationDirectSource.program application fits).compile
    (inputs fits geometry)).rowCount = _
  rw [OrdinarySourcePlan.Program.compile_rowCount]
  exact ApplicationDirectSource.sourceRows_length_eq_plan application

theorem rowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth application) → F)
    (encodes : Encodes geometry assignment source)
    (one : assignment (oneColumn geometry) = 1) :
    (plan fits geometry).RowsZero assignment ↔
      R1CS.RowsHold (sourceEnv source)
        (ApplicationDirectSource.sourceRows application) := by
  rw [plan, OrdinarySourcePlan.Program.rowsZero_iff
    (ApplicationDirectSource.program application fits) (inputs fits geometry)
    assignment (sourceEnv source) one
    (inputs_preserve fits geometry assignment source encodes)]
  exact ApplicationDirectSource.program_holds_iff_rowsHold application fits
    (sourceEnv source)

end NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
