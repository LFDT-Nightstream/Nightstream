import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram
import NightstreamFPrime.Layout.Stage1.PiDECSourceSupportData

/-!
Proves exact source custody for the compact running-transition ordinary-row
program. Each canonical transition source resolves through its Lean-authored
retained range or affine grid.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionRetainedBlocks
open RunningTransitionRetainedGeometry

private theorem stateRange_values (program : ApplicationProgram) :
    (stateRange program).sourceStart = 27810 ∧
      (stateRange program).sourceCount = 9 := by
  exact ⟨rfl, rfl⟩

private theorem outputRange_values (program : ApplicationProgram) :
    (outputRange program).sourceStart = 27819 ∧
      (outputRange program).sourceCount = 27819 := by
  exact ⟨rfl, rfl⟩

private theorem pointGrid_values (program : ApplicationProgram) :
    (PiCCSTranscriptOutputForms.pointGrid program).sourceStart = 5298316 ∧
      (PiCCSTranscriptOutputForms.pointGrid program).majorCount = 28 ∧
      (PiCCSTranscriptOutputForms.pointGrid program).majorSourceStride = 2192 := by
  simp only [PiCCSTranscriptOutputForms.pointGrid,
    SourceGrid.externalOfSemantic, SourceGrid.ofSemantic,
    PiCCSTranscriptOutputForms.pointSourceStart]
  exact ⟨rfl, rfl, rfl⟩

private theorem piDecRange_values (program : ApplicationProgram) :
    (piDecRange program).sourceStart = 11432356 ∧
      (piDecRange program).sourceCount = 31968 := by
  exact ⟨rfl, rfl⟩

private theorem freshRange_after_piDec (program : ApplicationProgram) :
    (piDecRange program).sourceStart + (piDecRange program).sourceCount ≤
      (freshRange program).sourceStart := by
  dsimp only [piDecRange, freshRange, SourceRange.ofSemantic]
  rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
    rw [RunningTransitionSourceSupport.piDecStart_eq]
    norm_num [Spartan.piCcsPhaseOffset])]
  simpa only [RunningTransitionSourceSupport.piDecStart,
    RunningTransitionSourceSupport.piDecCount, PiDECInputs.phaseOffset] using
    PiDECSourceSupport.mapped_logical_start_le_output

private theorem stateTarget (index : Fin RunningTransitionSourceSupport.stateCount) :
    Spartan.sourceToSpartan
        (RunningTransitionSourceSupport.stateStart + index.val) =
      27810 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPrivate]
  · rw [RunningTransitionSourceSupport.stateStart_eq]
    rfl
  · have bound := index.isLt
    change index.val < 9 at bound
    rw [RunningTransitionSourceSupport.stateStart_eq]
    norm_num [
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
    omega

private theorem outputTarget
    (index : Fin RunningTransitionSourceSupport.outputCount) :
    Spartan.sourceToSpartan
        (RunningTransitionSourceSupport.outputStart + index.val) =
      27819 + index.val := by
  have bound := index.isLt
  change index.val < 27819 at bound
  rw [RunningTransitionSourceSupport.outputStart_eq]
  unfold Spartan.sourceToSpartan
  rw [if_pos (by norm_num [Spartan.pilotSourceColumnCount]; omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by rw [PilotSpartan.priorPublicStart_value]; omega)]
  rw [if_neg (by rw [PilotSpartan.outputPreimageStart_value]; omega)]
  rw [if_pos (by rw [PilotSpartan.outputDigestStart_value]; omega)]
  unfold Spartan.liftPilotColumn
  rw [if_pos (by
    rw [PilotSpartan.secondPrivateStart_value,
      PilotSpartan.outputPreimageStart_value]
    norm_num [Spartan.pilotInputPrivateColumnCount]
    omega)]
  rw [PilotSpartan.secondPrivateStart_value,
    PilotSpartan.outputPreimageStart_value]
  omega

private theorem roundTarget
    (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    Spartan.sourceToSpartan
        (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      5298316 + coordinate.val * 2192 + component.val := by
  have grouped : PiCCSTranscriptOutputForms.pointSource coordinate component =
      PiCCSTranscriptOutputForms.pointSourceStart +
        (coordinate.val * RunningTransitionInputs.roundStride + component.val) := by
    unfold PiCCSTranscriptOutputForms.pointSource
    omega
  rw [grouped, Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
    norm_num [PiCCSTranscriptOutputForms.pointSourceStart,
      PiCCSStarts.roundTranscriptWitnessStart_eq,
      RunningTransitionInputs.roundSampleC0Offset, Spartan.piCcsPhaseOffset])]
  have start : Spartan.sourceToSpartan
      PiCCSTranscriptOutputForms.pointSourceStart = 5298316 := rfl
  rw [start]
  norm_num [RunningTransitionInputs.roundStride]
  omega

private theorem piDecTarget
    (index : Fin RunningTransitionSourceSupport.piDecCount) :
    Spartan.sourceToSpartan
        (RunningTransitionSourceSupport.piDecStart + index.val) =
      Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart +
        index.val := by
  exact Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
    rw [RunningTransitionSourceSupport.piDecStart_eq]
    norm_num [Spartan.piCcsPhaseOffset])

private theorem freshTarget (index : Fin freshCount) :
    Spartan.sourceToSpartan
        (RunningTransitionInputs.phaseOffset + index.val) =
      Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset + index.val := by
  exact Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
    exact Nat.le_trans (by decide : Spartan.piCcsPhaseOffset ≤ PiDECInputs.phaseOffset)
      RunningTransitionInputs.piDecPhaseOffset_le)

theorem stateRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.stateCount) :
    (stateRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (RunningTransitionSourceSupport.stateStart + index.val)) =
      some ((RunningTransitionDirectPlan.Location.state index).form geometry) := by
  rw [stateTarget]
  simpa [stateRange, RunningTransitionDirectPlan.Location.form] using!
    (SourceRange.form?_ofSemantic (stateBlock program) (stateStart program)
      (Spartan.sourceToSpartan RunningTransitionSourceSupport.stateStart)
      RunningTransitionSourceSupport.stateCount 0
      (stateFits geometry) (by rfl) index)

theorem outputRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.outputCount) :
    (outputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (RunningTransitionSourceSupport.outputStart + index.val)) =
      some ((RunningTransitionDirectPlan.Location.output index).form geometry) := by
  rw [outputTarget]
  simpa [outputRange, RunningTransitionDirectPlan.Location.form] using!
    (SourceRange.form?_ofSemantic (outputBlock program) (outputStart program)
      (Spartan.sourceToSpartan RunningTransitionSourceSupport.outputStart)
      RunningTransitionSourceSupport.outputCount 0
      (outputFits geometry) (by rfl) index)

theorem roundC0_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) :
    (PiCCSTranscriptOutputForms.pointGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC0Offset)) =
      some ((RunningTransitionDirectPlan.Location.roundC0 coordinate).form
        geometry) := by
  rw [← PiCCSTranscriptOutputForms.pointSource_c0]
  exact PiCCSTranscriptOutputForms.pointGrid_form? (poseidonGeometry geometry)
    coordinate 0

theorem roundC1_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) :
    (PiCCSTranscriptOutputForms.pointGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC1Offset)) =
      some ((RunningTransitionDirectPlan.Location.roundC1 coordinate).form
        geometry) := by
  rw [← PiCCSTranscriptOutputForms.pointSource_c1]
  exact PiCCSTranscriptOutputForms.pointGrid_form? (poseidonGeometry geometry)
    coordinate 1

theorem piDecRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.piDecCount) :
    (piDecRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (RunningTransitionSourceSupport.piDecStart + index.val)) =
      some ((RunningTransitionDirectPlan.Location.piDec index).form geometry) := by
  rw [piDecTarget]
  simpa [piDecRange, RunningTransitionDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (piDecBlock program) (piDecStart program)
      (Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart)
      RunningTransitionSourceSupport.piDecCount 0
      (piDecFits geometry) (by rfl) index)

theorem freshRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin freshCount) :
    (freshRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (RunningTransitionInputs.phaseOffset + index.val)) =
      some ((RunningTransitionDirectPlan.Location.fresh index).form geometry) := by
  rw [freshTarget]
  simpa [freshRange, RunningTransitionDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshBlock program) (freshStart program)
      (Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset)
      freshCount 0 (freshFits geometry) (by rfl) index)

private theorem piDecMappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart =
      11432356 := by
  exact (piDecRange_values program).1

private theorem freshMappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset =
      (freshRange program).sourceStart := by
  simp only [freshRange, SourceRange.ofSemantic]

/-- The compact substitution reconstructs every direct running-transition
source location and rejects all overlapping interpretations. -/
theorem substitution_location_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : RunningTransitionDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  rcases stateRange_values program with ⟨stateStartValue, stateCountValue⟩
  rcases outputRange_values program with
    ⟨outputStartValue, outputCountValue⟩
  rcases pointGrid_values program with
    ⟨pointStartValue, pointCountValue, pointStrideValue⟩
  rcases piDecRange_values program with
    ⟨piDecStartValue, piDecCountValue⟩
  have piDecFresh := freshRange_after_piDec program
  rw [piDecStartValue, piDecCountValue] at piDecFresh
  cases location with
  | state index =>
      have indexBound := index.isLt
      change index.val < 9 at indexBound
      have selected := stateRange_form? geometry index
      rw [stateTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [stateTarget]
      have outputNone := SourceRange.form?_eq_none_of_before
        (outputRange program) logicalWidth (27810 + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth (27810 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (27810 + index.val) (by omega)
      have pointNone := SourceGrid.form?_eq_none_of_before
        (PiCCSTranscriptOutputForms.pointGrid program) logicalWidth
          (27810 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, selected, outputNone,
        piDecNone, freshNone, pointNone]
  | output index =>
      have indexBound := index.isLt
      change index.val < 27819 at indexBound
      have selected := outputRange_form? geometry index
      rw [outputTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [outputTarget]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth (27819 + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth (27819 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (27819 + index.val) (by omega)
      have pointNone := SourceGrid.form?_eq_none_of_before
        (PiCCSTranscriptOutputForms.pointGrid program) logicalWidth
          (27819 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, stateNone, selected,
        piDecNone, freshNone, pointNone]
  | roundC0 coordinate =>
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      have selected := roundC0_form? geometry coordinate
      rw [← PiCCSTranscriptOutputForms.pointSource_c0, roundTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [← PiCCSTranscriptOutputForms.pointSource_c0, roundTarget]
      simp only [Fin.val_zero, Nat.add_zero] at selected ⊢
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth
          (5298316 + coordinate.val * 2192) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth
          (5298316 + coordinate.val * 2192) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth
          (5298316 + coordinate.val * 2192) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth
          (5298316 + coordinate.val * 2192) (by omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, freshNone, selected]
  | roundC1 coordinate =>
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      have selected := roundC1_form? geometry coordinate
      rw [← PiCCSTranscriptOutputForms.pointSource_c1, roundTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [← PiCCSTranscriptOutputForms.pointSource_c1, roundTarget]
      simp only [Fin.val_one] at selected ⊢
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth
          (5298316 + coordinate.val * 2192 + 1) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth
          (5298316 + coordinate.val * 2192 + 1) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth
          (5298316 + coordinate.val * 2192 + 1) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth
          (5298316 + coordinate.val * 2192 + 1) (by omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, freshNone, selected]
  | piDec index =>
      have indexBound := index.isLt
      change index.val < 31968 at indexBound
      have selected := piDecRange_form? geometry index
      rw [piDecTarget, piDecMappedStart program] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [piDecTarget, piDecMappedStart program]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth (11432356 + index.val) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth (11432356 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (11432356 + index.val) (by omega)
      have pointNone := SourceGrid.form?_eq_none_of_after
        (PiCCSTranscriptOutputForms.pointGrid program) logicalWidth
          (11432356 + index.val)
        (by rw [pointStrideValue]; omega)
        (by rw [pointStartValue, pointCountValue, pointStrideValue]; omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        selected, freshNone, pointNone]
  | fresh index =>
      have selected := freshRange_form? geometry index
      rw [freshTarget, freshMappedStart program] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [freshTarget, freshMappedStart program]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth
          ((freshRange program).sourceStart + index.val) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth
          ((freshRange program).sourceStart + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_after
        (piDecRange program) logicalWidth
          ((freshRange program).sourceStart + index.val) (by omega)
      have pointNone := SourceGrid.form?_eq_none_of_after
        (PiCCSTranscriptOutputForms.pointGrid program) logicalWidth
          ((freshRange program).sourceStart + index.val)
        (by rw [pointStrideValue]; omega)
        (by rw [pointStartValue, pointCountValue, pointStrideValue]; omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, selected, pointNone]

/-- On every source column used by a canonical running-transition row, the
package substitution is exactly the direct Lean source map. -/
theorem substitution_agrees_on_target
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : RunningTransitionSourceSupport.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((RunningTransitionDirectPlan.sourceMap geometry).form column) := by
  rcases RunningTransitionDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, mapped⟩
  change (substitution program).form? logicalWidth column.val =
    some (match RunningTransitionDirectPlan.classifyTarget column.val with
      | none => .empty
      | some value => value.location.form geometry)
  rw [found]
  have target :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

end NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram
