import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixProgram
import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgramSubstitution
import NightstreamFPrime.Layout.Stage1.RunningTransitionReducedRows
import NightstreamFPrime.Layout.Stage1.RunningTransitionValues

/-!
Connects the reduced matrix program to the unchanged running-transition
source values. The source environment is decoded from actual retained forms;
no caller supplies arithmetic equalities for those values.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.MatrixProgram.AffineGrid
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionReducedMatrixProgram RunningTransitionRetainedBlocks

abbrev Geometry := RunningTransitionRetainedGeometry.Geometry
abbrev Location := RunningTransitionDirectPlan.Location

def flagFits {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (flagWire program).start + (flagWire program).coordinateCount ≤ logicalWidth := by
  have fits := RunningTransitionRetainedGeometry.freshFits geometry
  change RunningTransitionRetainedGeometry.freshStart program + 296138 * 41 ≤ logicalWidth at fits
  change RunningTransitionRetainedGeometry.freshStart program + 41 + 1 ≤ logicalWidth
  omega

def inverseFits {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (inverseWire program).start + (inverseWire program).coordinateCount ≤ logicalWidth := by
  have fits := RunningTransitionRetainedGeometry.freshFits geometry
  change RunningTransitionRetainedGeometry.freshStart program + 296138 * 41 ≤ logicalWidth at fits
  change RunningTransitionRetainedGeometry.freshStart program + 41 ≤ logicalWidth
  omega

def wireForm {logicalWidth : Nat} (wire : RetainedBlock)
    (fits : wire.start + wire.coordinateCount ≤ logicalWidth)
    (slot : Fin wire.slotCount) : SparseForm logicalWidth :=
  wire.semantic.form wire.start fits slot

theorem wireForm_ofSemantic {sourceWidth logicalWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ logicalWidth) (slot : Fin block.slotCount) :
    wireForm (RetainedBlock.ofSemantic block start) fits slot = block.form start fits slot := by
  rfl

def flagForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : SparseForm logicalWidth :=
  wireForm (flagWire program) (flagFits geometry) ⟨0, by change 0 < 1; omega⟩

def inverseForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : SparseForm logicalWidth :=
  wireForm (inverseWire program) (inverseFits geometry) ⟨0, by change 0 < 1; omega⟩

/-- Reuse the existing authoritative source substitution. Only the new flag
changes its representation from the old scratch field to one Boolean cell. -/
def sourceForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (column : Nat) : SparseForm logicalWidth :=
  if column = RunningTransitionReducedRows.flagIndex then flagForm geometry else
    ((RunningTransitionMatrixProgram.substitution program).form? logicalWidth
      (Spartan.sourceToSpartan column)).getD .empty

def decodedEnv {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (assignment : Assignment F logicalWidth) : Env :=
  fun column => (sourceForm geometry column).eval assignment

theorem sourceForm_location {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (location : Location)
    (notFlag : location.sourceColumn ≠ RunningTransitionReducedRows.flagIndex) :
    sourceForm geometry location.sourceColumn = location.form geometry := by
  rw [sourceForm, if_neg notFlag,
    RunningTransitionMatrixProgram.substitution_location_form? geometry location]
  rfl

theorem sourceForm_state {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.stateCount) :
    sourceForm geometry (RunningTransitionDirectPlan.Location.state index).sourceColumn =
      (RunningTransitionDirectPlan.Location.state index).form geometry := by
  apply sourceForm_location
  have bound := index.isLt
  change index.val < 11 at bound
  simp only [RunningTransitionDirectPlan.Location.sourceColumn,
    RunningTransitionSourceSupport.stateStart_eq, RunningTransitionReducedRows.flagIndex,
    RunningTransitionLayout.logicalColumnCount_eq]
  omega

theorem sourceForm_output {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.outputCount) :
    sourceForm geometry (RunningTransitionDirectPlan.Location.output index).sourceColumn =
      (RunningTransitionDirectPlan.Location.output index).form geometry := by
  apply sourceForm_location
  have bound := index.isLt
  change index.val < 49393 at bound
  simp only [RunningTransitionDirectPlan.Location.sourceColumn,
    RunningTransitionSourceSupport.outputStart_eq, RunningTransitionReducedRows.flagIndex,
    RunningTransitionLayout.logicalColumnCount_eq]
  omega

theorem sourceForm_piDec {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.piDecCount) :
    sourceForm geometry (RunningTransitionDirectPlan.Location.piDec index).sourceColumn =
      (RunningTransitionDirectPlan.Location.piDec index).form geometry := by
  apply sourceForm_location
  have bound := index.isLt
  change index.val < 49248 at bound
  simp only [RunningTransitionDirectPlan.Location.sourceColumn,
    RunningTransitionSourceSupport.piDecStart_eq, RunningTransitionReducedRows.flagIndex,
    RunningTransitionLayout.logicalColumnCount_eq]
  omega

theorem sourceForm_point {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin Lifecycle.productionShape.cubeVariables) (component : Fin 2) :
    sourceForm geometry (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate component := by
  have bound := coordinate.isLt
  change coordinate.val < 28 at bound
  fin_cases component
  · change sourceForm geometry (PiCCSTranscriptOutputForms.pointSource coordinate (0 : Fin 2)) =
      PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate (0 : Fin 2)
    rw [PiCCSTranscriptOutputForms.pointSource_c0]
    apply sourceForm_location geometry (RunningTransitionDirectPlan.Location.roundC0 coordinate)
    simp only [RunningTransitionDirectPlan.Location.sourceColumn,
      RunningTransitionReducedRows.flagIndex, RunningTransitionLayout.logicalColumnCount_eq,
      PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [RunningTransitionInputs.roundStride, RunningTransitionInputs.roundSampleC0Offset]
    omega
  · change sourceForm geometry (PiCCSTranscriptOutputForms.pointSource coordinate (1 : Fin 2)) =
      PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate (1 : Fin 2)
    rw [PiCCSTranscriptOutputForms.pointSource_c1]
    apply sourceForm_location geometry (RunningTransitionDirectPlan.Location.roundC1 coordinate)
    simp only [RunningTransitionDirectPlan.Location.sourceColumn,
      RunningTransitionReducedRows.flagIndex, RunningTransitionLayout.logicalColumnCount_eq,
      PiCCSStarts.roundTranscriptWitnessStart_eq]
    norm_num [RunningTransitionInputs.roundStride, RunningTransitionInputs.roundSampleC1Offset]
    omega

theorem sourceForm_flag {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    sourceForm geometry RunningTransitionReducedRows.flagIndex = flagForm geometry := by
  simp only [sourceForm, ↓reduceIte]

theorem inverseForm_eq_source {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    inverseForm geometry = sourceForm geometry RunningTransitionInputs.phaseOffset := by
  have old := sourceForm_location geometry (RunningTransitionDirectPlan.Location.fresh ⟨0, by rw [freshCount_eq]; omega⟩) (by
    change RunningTransitionInputs.phaseOffset + 0 ≠
      RunningTransitionInputs.phaseOffset + 1
    omega)
  change sourceForm geometry RunningTransitionInputs.phaseOffset = _ at old
  rw [old]
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · rfl

theorem retainedRule_form {logicalWidth : Nat} (selected : Region)
    (major : Fin selected.majorCount) (middle : Fin selected.middleCount)
    (minor : Fin selected.minorCount) (wire : RetainedBlock)
    (fits : wire.start + wire.coordinateCount ≤ logicalWidth)
    (oneColumn slot majorStride middleStride minorStride : Nat) (coefficient : F)
    (bound : slot + major.val * majorStride + middle.val * middleStride +
      minor.val * minorStride < wire.slotCount) :
    (retained selected wire slot majorStride middleStride minorStride coefficient).form?
      logicalWidth oneColumn
        ⟨selected.majorStart + major.val, selected.middleStart + middle.val,
          selected.minorStart + minor.val⟩ =
      some (some (applyCoefficient coefficient (wireForm wire fits
        ⟨slot + major.val * majorStride + middle.val * middleStride + minor.val * minorStride,
          bound⟩))) := by
  exact Rule.retained_form?_ofSemantic selected major middle minor wire.semantic wire.start
    fits oneColumn slot majorStride middleStride minorStride coefficient bound

theorem flagProgram_form {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (selected : Region)
    (major : Fin selected.majorCount) (middle : Fin selected.middleCount)
    (minor : Fin selected.minorCount) (oneColumn : Nat) :
    (flagProgram program selected).form? logicalWidth oneColumn
      ⟨selected.majorStart + major.val, selected.middleStart + middle.val,
        selected.minorStart + minor.val⟩ = some (flagForm geometry) := by
  apply AffineGrid.Program.singleton_form?_of_selected
  have loaded := retainedRule_form selected major middle minor
    (flagWire program) (flagFits geometry) oneColumn 0 0 0 0 (1 : F) (by simp only [Nat.mul_zero, Nat.zero_add]; change 0 < 1; decide)
  simpa only [Nat.mul_zero, Nat.zero_add, applyCoefficient, if_pos rfl, flagForm] using! loaded

theorem baseProgram_form {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (selected : Region)
    (major : Fin selected.majorCount) (middle : Fin selected.middleCount)
    (minor : Fin selected.minorCount) (oneColumn : Fin logicalWidth) :
    (baseProgram program selected).form? logicalWidth oneColumn.val
      ⟨selected.majorStart + major.val, selected.middleStart + middle.val,
        selected.minorStart + minor.val⟩ =
      some (SparseForm.add (SparseForm.singleton oneColumn 1)
        (SparseForm.scale (-1) (flagForm geometry))) := by
  have one := Rule.constant_form? selected major middle minor oneColumn (1 : F)
  have flag := retainedRule_form selected major middle minor
    (flagWire program) (flagFits geometry) oneColumn.val 0 0 0 0 (-1) (by simp only [Nat.mul_zero, Nat.zero_add]; change 0 < 1; decide)
  have negative : (-1 : F) ≠ 1 := by decide
  have loaded := AffineGrid.Program.two_form?_of_results
    (constant selected 1) (retained selected (flagWire program) 0 0 0 0 (-1))
    oneColumn.val
    ⟨selected.majorStart + major.val, selected.middleStart + middle.val,
      selected.minorStart + minor.val⟩ _ _ one flag
  simpa only [baseProgram, Nat.mul_zero, Nat.zero_add, applyCoefficient, if_neg negative,
    addSelected, flagForm, SparseForm.add, SparseForm.empty, List.nil_append] using loaded

def flagForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := (RunningTransitionDirectPlan.Location.state ⟨0, by change 0 < 11; omega⟩).form geometry
    b := inverseForm geometry
    c := flagForm geometry }

def bindingForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := (RunningTransitionDirectPlan.Location.state ⟨0, by change 0 < 11; omega⟩).form geometry
    b := SparseForm.add (SparseForm.singleton oneColumn 1)
      (SparseForm.scale (-1) (flagForm geometry))
    c := .empty }

theorem flagGrid_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    (flagGrid program oneColumn.val).row? logicalWidth 0 = some {
      selector := SparseForm.singleton oneColumn 1
      a := (RunningTransitionDirectPlan.Location.state ⟨0, by change 0 < 11; omega⟩).form geometry
      b := inverseForm geometry
      c := flagForm geometry } := by
  let selected := region 1 1 0 1
  have left := retainedRule_form selected ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ (stateWire program)
    (RunningTransitionRetainedGeometry.stateFits geometry) oneColumn.val 0 0 0 0 (1 : F)
    (by simp [stateWire, RetainedBlock.ofSemantic])
  have right := retainedRule_form selected ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ (inverseWire program)
    (inverseFits geometry) oneColumn.val 0 0 0 0 (1 : F) (by simp only [Nat.mul_zero, Nat.zero_add]; change 0 < 1; decide)
  apply MultiplicationGrid.Block.row?_of_results (flagGrid program oneColumn.val)
    oneColumn rfl ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩
  · apply AffineGrid.Program.singleton_form?_of_selected
    simpa only [selected, Nat.mul_zero, Nat.zero_add, applyCoefficient, if_pos rfl,
      stateWire, wireForm, RetainedBlock.ofSemantic, RetainedBlock.semantic,
      RunningTransitionDirectPlan.Location.form] using! left
  · apply AffineGrid.Program.singleton_form?_of_selected
    simpa only [selected, Nat.mul_zero, Nat.zero_add, applyCoefficient, if_pos rfl,
      inverseForm] using! right
  · exact flagProgram_form geometry selected ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ oneColumn.val

theorem bindingGrid_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    (bindingGrid program oneColumn.val).row? logicalWidth 0 = some {
      selector := SparseForm.singleton oneColumn 1
      a := (RunningTransitionDirectPlan.Location.state ⟨0, by change 0 < 11; omega⟩).form geometry
      b := SparseForm.add (SparseForm.singleton oneColumn 1)
        (SparseForm.scale (-1) (flagForm geometry))
      c := .empty } := by
  let selected := region 1 1 0 1
  have left := retainedRule_form selected ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ (stateWire program)
    (RunningTransitionRetainedGeometry.stateFits geometry) oneColumn.val 0 0 0 0 (1 : F)
    (by simp [stateWire, RetainedBlock.ofSemantic])
  apply MultiplicationGrid.Block.row?_of_results (bindingGrid program oneColumn.val)
    oneColumn rfl ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩
  · apply AffineGrid.Program.singleton_form?_of_selected
    simpa only [selected, Nat.mul_zero, Nat.zero_add, applyCoefficient, if_pos rfl,
      stateWire, wireForm, RetainedBlock.ofSemantic, RetainedBlock.semantic,
      RunningTransitionDirectPlan.Location.form] using! left
  · exact baseProgram_form geometry selected ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ oneColumn
  · rfl

theorem flagForms_preserve {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1) :
    (flagForms geometry oneColumn).Preserves assignment (decodedEnv geometry assignment)
      RunningTransitionReducedRows.flagRow := by
  have state := congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
    (sourceForm_state geometry ⟨0, by change 0 < 11; omega⟩)
  have inverse := congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
    (inverseForm_eq_source geometry)
  have flag := congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
    (sourceForm_flag geometry)
  refine ⟨by simp [flagForms, one], ?_, ?_, ?_⟩
  · change ((RunningTransitionDirectPlan.Location.state ⟨0, by change 0 < 11; omega⟩).form geometry).eval assignment =
      (R1CS.LinearCombination.ofVar
        (PilotProduction.priorPreimageStart + RunningTransitionInputs.iterationWordIndex)).eval
          (decodedEnv geometry assignment)
    rw [R1CS.LinearCombination.eval_ofVar]
    exact state.symm
  · change (inverseForm geometry).eval assignment =
      (R1CS.LinearCombination.ofVar RunningTransitionInputs.phaseOffset).eval
        (decodedEnv geometry assignment)
    rw [R1CS.LinearCombination.eval_ofVar]
    exact inverse
  · change (flagForm geometry).eval assignment =
      (R1CS.LinearCombination.ofVar RunningTransitionReducedRows.flagIndex).eval
        (decodedEnv geometry assignment)
    rw [R1CS.LinearCombination.eval_ofVar]
    exact flag.symm

theorem bindingForms_preserve {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1) :
    (bindingForms geometry oneColumn).Preserves assignment (decodedEnv geometry assignment)
      RunningTransitionReducedRows.bindingRow := by
  have flag := flagForms_preserve geometry oneColumn assignment one
  refine ⟨by simp [bindingForms, one], flag.2.1, ?_, ?_⟩
  · change (SparseForm.add (SparseForm.singleton oneColumn 1)
        (SparseForm.scale (-1) (flagForm geometry))).eval assignment =
      (R1CS.LinearCombination.add R1CS.LinearCombination.one
        (R1CS.LinearCombination.scale (-1)
          (R1CS.LinearCombination.ofVar RunningTransitionReducedRows.flagIndex))).eval
            (decodedEnv geometry assignment)
    have flagValue := flag.2.2.2
    change (flagForm geometry).eval assignment =
      (R1CS.LinearCombination.ofVar RunningTransitionReducedRows.flagIndex).eval
        (decodedEnv geometry assignment) at flagValue
    rw [R1CS.LinearCombination.eval_ofVar] at flagValue
    simp only [SparseForm.add_eval, SparseForm.singleton_eval, one, mul_one,
      SparseForm.scale_eval, R1CS.LinearCombination.eval_add,
      R1CS.LinearCombination.eval_one, R1CS.LinearCombination.eval_scale,
      R1CS.LinearCombination.eval_ofVar]
    rw [flagValue]
  · change SparseForm.empty.eval assignment =
      R1CS.LinearCombination.zero.eval (decodedEnv geometry assignment)
    simp only [SparseForm.empty_eval, R1CS.LinearCombination.eval_zero]

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics
