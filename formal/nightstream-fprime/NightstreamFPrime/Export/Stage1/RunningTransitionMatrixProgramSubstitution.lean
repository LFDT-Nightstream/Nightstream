import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

/-!
Proves exact source custody for the compact running-transition ordinary-row
program. Each canonical transition source resolves through its Lean-authored
retained range or affine grid.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionRetainedBlocks
open RunningTransitionRetainedGeometry

private theorem stateRange_values (program : ApplicationProgram) :
    (stateRange program).sourceStart = 28 ∧
      (stateRange program).sourceCount = 11 := by
  exact ⟨rfl, rfl⟩

private theorem outputRange_values (program : ApplicationProgram) :
    (outputRange program).sourceStart = 49393 ∧
      (outputRange program).sourceCount = 49393 := by
  exact ⟨rfl, rfl⟩

private theorem roundC0Grid_values (program : ApplicationProgram) :
    (roundC0Grid program).sourceStart = 15031534 ∧
      (roundC0Grid program).majorCount = 28 ∧
      (roundC0Grid program).majorSourceStride = 5328 := by
  exact ⟨rfl, rfl, rfl⟩

private theorem roundC1Grid_values (program : ApplicationProgram) :
    (roundC1Grid program).sourceStart = 15032126 ∧
      (roundC1Grid program).majorCount = 28 ∧
      (roundC1Grid program).majorSourceStride = 5328 := by
  exact ⟨rfl, rfl, rfl⟩

private theorem piDecRange_values (program : ApplicationProgram) :
    (piDecRange program).sourceStart = 28972970 ∧
      (piDecRange program).sourceCount = 49248 := by
  exact ⟨rfl, rfl⟩

private theorem freshRange_values (program : ApplicationProgram) :
    (freshRange program).sourceStart = 29040308 ∧
      (freshRange program).sourceCount = 296138 := by
  exact ⟨rfl, rfl⟩

private theorem stateTarget (index : Fin RunningTransitionSourceSupport.stateCount) :
    Spartan.sourceToSpartan
        (RunningTransitionSourceSupport.stateStart + index.val) =
      28 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPrivate]
  · rw [RunningTransitionSourceSupport.stateStart_eq]
    rfl
  · have bound := index.isLt
    change index.val < 11 at bound
    rw [RunningTransitionSourceSupport.stateStart_eq]
    norm_num [
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
    omega

private theorem outputTarget
    (index : Fin RunningTransitionSourceSupport.outputCount) :
    Spartan.sourceToSpartan
        (RunningTransitionSourceSupport.outputStart + index.val) =
      49393 + index.val := by
  have bound := index.isLt
  change index.val < 49393 at bound
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

private theorem roundC0Target
    (coordinate : Fin productionShape.cubeVariables) :
    Spartan.sourceToSpartan
        (PiCCSStarts.roundTranscriptWitnessStart +
          coordinate.val * RunningTransitionInputs.roundStride +
            RunningTransitionInputs.roundSampleC0Offset) =
      Spartan.sourceToSpartan roundC0SourceStart +
        coordinate.val * RunningTransitionInputs.roundStride := by
  calc
    _ = Spartan.sourceToSpartan
        (roundC0SourceStart +
          coordinate.val * RunningTransitionInputs.roundStride) := by
      apply congrArg Spartan.sourceToSpartan
      rw [roundC0SourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundSampleC0Offset]
      omega
    _ = _ := Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
      rw [roundC0SourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundSampleC0Offset,
        Spartan.piCcsPhaseOffset])

private theorem roundC1Target
    (coordinate : Fin productionShape.cubeVariables) :
    Spartan.sourceToSpartan
        (PiCCSStarts.roundTranscriptWitnessStart +
          coordinate.val * RunningTransitionInputs.roundStride +
            RunningTransitionInputs.roundSampleC1Offset) =
      Spartan.sourceToSpartan roundC1SourceStart +
        coordinate.val * RunningTransitionInputs.roundStride := by
  calc
    _ = Spartan.sourceToSpartan
        (roundC1SourceStart +
          coordinate.val * RunningTransitionInputs.roundStride) := by
      apply congrArg Spartan.sourceToSpartan
      rw [roundC1SourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundSampleC1Offset]
      omega
    _ = _ := Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
      rw [roundC1SourceStart, PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundSampleC1Offset,
        Spartan.piCcsPhaseOffset])

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
    norm_num [RunningTransitionInputs.phaseOffset, Spartan.piCcsPhaseOffset])

theorem stateRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin RunningTransitionSourceSupport.stateCount) :
    (stateRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (RunningTransitionSourceSupport.stateStart + index.val)) =
      some ((RunningTransitionDirectPlan.Location.state index).form geometry) := by
  rw [stateTarget]
  simpa [stateRange, RunningTransitionDirectPlan.Location.form] using
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
  simpa [outputRange, RunningTransitionDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (outputBlock program) (outputStart program)
      (Spartan.sourceToSpartan RunningTransitionSourceSupport.outputStart)
      RunningTransitionSourceSupport.outputCount 0
      (outputFits geometry) (by rfl) index)

theorem roundC0Grid_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) :
    (roundC0Grid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC0Offset)) =
      some ((RunningTransitionDirectPlan.Location.roundC0 coordinate).form
        geometry) := by
  rw [roundC0Target]
  simpa [roundC0Grid, RunningTransitionDirectPlan.Location.form] using
    (SourceGrid.form?_ofSemantic (roundC0Block program) (roundC0Start program)
      (Spartan.sourceToSpartan roundC0SourceStart)
      productionShape.cubeVariables RunningTransitionInputs.roundStride
      1 1 1 0 1 0 (roundC0Fits geometry)
      (by norm_num [RunningTransitionInputs.roundStride]) (by norm_num)
      coordinate ⟨0, by omega⟩ ⟨0, by omega⟩
      (by norm_num [RunningTransitionInputs.roundStride]) (by norm_num)
      (by
        have bound := coordinate.isLt
        change coordinate.val < 28 at bound
        simpa only [Nat.zero_add, Nat.mul_one, Nat.mul_zero, Nat.add_zero,
          roundC0Block_slotCount] using bound))

theorem roundC1Grid_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) :
    (roundC1Grid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC1Offset)) =
      some ((RunningTransitionDirectPlan.Location.roundC1 coordinate).form
        geometry) := by
  rw [roundC1Target]
  simpa [roundC1Grid, RunningTransitionDirectPlan.Location.form] using
    (SourceGrid.form?_ofSemantic (roundC1Block program) (roundC1Start program)
      (Spartan.sourceToSpartan roundC1SourceStart)
      productionShape.cubeVariables RunningTransitionInputs.roundStride
      1 1 1 0 1 0 (roundC1Fits geometry)
      (by norm_num [RunningTransitionInputs.roundStride]) (by norm_num)
      coordinate ⟨0, by omega⟩ ⟨0, by omega⟩
      (by norm_num [RunningTransitionInputs.roundStride]) (by norm_num)
      (by
        have bound := coordinate.isLt
        change coordinate.val < 28 at bound
        simpa only [Nat.zero_add, Nat.mul_one, Nat.mul_zero, Nat.add_zero,
          roundC1Block_slotCount] using bound))

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

private theorem roundC0MappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan roundC0SourceStart = 15031534 := by
  exact (roundC0Grid_values program).1

private theorem roundC1MappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan roundC1SourceStart = 15032126 := by
  exact (roundC1Grid_values program).1

private theorem piDecMappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan RunningTransitionSourceSupport.piDecStart =
      28972970 := by
  exact (piDecRange_values program).1

private theorem freshMappedStart (program : ApplicationProgram) :
    Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset = 29040308 := by
  exact (freshRange_values program).1

private theorem roundC1Grid_form?_none_at_roundC0
    {program : ApplicationProgram} {logicalWidth : Nat}
    (coordinate : Fin productionShape.cubeVariables) :
    (roundC1Grid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC0Offset)) = none := by
  rw [roundC0Target, roundC0MappedStart program]
  change (roundC1Grid program).form? logicalWidth
    (15031534 + coordinate.val * 5328) = none
  have coordinateBound := coordinate.isLt
  change coordinate.val < 28 at coordinateBound
  by_cases first : coordinate.val = 0
  · apply SourceGrid.form?_eq_none_of_before
    rw [(roundC1Grid_values program).1]
    omega
  · have previousBound : coordinate.val - 1 <
        (roundC1Grid program).majorCount := by
      rw [(roundC1Grid_values program).2.1]
      omega
    let previous : Fin (roundC1Grid program).majorCount :=
      ⟨coordinate.val - 1, previousBound⟩
    have rejected := SourceGrid.form?_eq_none_at_minorAfter
      (roundC1Grid program) logicalWidth previous 4736 0
      (by
        rw [(roundC1Grid_values program).2.2]
        omega)
      (by
        change 0 < 1
        omega)
      (by
        change 4736 * 1 + 0 < 5328
        omega)
      (by
        change 0 < 1
        omega)
      (by
        change 1 ≤ 4736
        omega)
    have sourceEq :
        (roundC1Grid program).sourceStart +
              previous.val * (roundC1Grid program).majorSourceStride +
              4736 * (roundC1Grid program).minorSourceStride + 0 =
          15031534 + coordinate.val * 5328 := by
      change 15032126 + (coordinate.val - 1) * 5328 + 4736 * 1 + 0 =
        15031534 + coordinate.val * 5328
      omega
    rw [← sourceEq]
    exact rejected

private theorem roundC0Grid_form?_none_at_roundC1
    {program : ApplicationProgram} {logicalWidth : Nat}
    (coordinate : Fin productionShape.cubeVariables) :
    (roundC0Grid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride +
              RunningTransitionInputs.roundSampleC1Offset)) = none := by
  rw [roundC1Target, roundC1MappedStart program]
  change (roundC0Grid program).form? logicalWidth
    (15032126 + coordinate.val * 5328) = none
  have coordinateBound := coordinate.isLt
  change coordinate.val < 28 at coordinateBound
  have majorBound : coordinate.val < (roundC0Grid program).majorCount := by
    rw [(roundC0Grid_values program).2.1]
    exact coordinateBound
  let major : Fin (roundC0Grid program).majorCount :=
    ⟨coordinate.val, majorBound⟩
  have rejected := SourceGrid.form?_eq_none_at_minorAfter
    (roundC0Grid program) logicalWidth major 592 0
    (by
      rw [(roundC0Grid_values program).2.2]
      omega)
    (by
      change 0 < 1
      omega)
    (by
      change 592 * 1 + 0 < 5328
      omega)
    (by
      change 0 < 1
      omega)
    (by
      change 1 ≤ 592
      omega)
  have sourceEq :
      (roundC0Grid program).sourceStart +
            major.val * (roundC0Grid program).majorSourceStride +
            592 * (roundC0Grid program).minorSourceStride + 0 =
        15032126 + coordinate.val * 5328 := by
    change 15031534 + coordinate.val * 5328 + 592 * 1 + 0 =
      15032126 + coordinate.val * 5328
    omega
  rw [← sourceEq]
  exact rejected

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
  rcases roundC0Grid_values program with
    ⟨c0StartValue, c0CountValue, c0StrideValue⟩
  rcases roundC1Grid_values program with
    ⟨c1StartValue, c1CountValue, c1StrideValue⟩
  rcases piDecRange_values program with
    ⟨piDecStartValue, piDecCountValue⟩
  rcases freshRange_values program with
    ⟨freshStartValue, freshCountValue⟩
  cases location with
  | state index =>
      have indexBound := index.isLt
      change index.val < 11 at indexBound
      have selected := stateRange_form? geometry index
      rw [stateTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [stateTarget]
      have outputNone := SourceRange.form?_eq_none_of_before
        (outputRange program) logicalWidth (28 + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth (28 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (28 + index.val) (by omega)
      have c0None := SourceGrid.form?_eq_none_of_before
        (roundC0Grid program) logicalWidth (28 + index.val) (by omega)
      have c1None := SourceGrid.form?_eq_none_of_before
        (roundC1Grid program) logicalWidth (28 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, selected, outputNone,
        piDecNone, freshNone, c0None, c1None]
  | output index =>
      have indexBound := index.isLt
      change index.val < 49393 at indexBound
      have selected := outputRange_form? geometry index
      rw [outputTarget] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [outputTarget]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth (49393 + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth (49393 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (49393 + index.val) (by omega)
      have c0None := SourceGrid.form?_eq_none_of_before
        (roundC0Grid program) logicalWidth (49393 + index.val) (by omega)
      have c1None := SourceGrid.form?_eq_none_of_before
        (roundC1Grid program) logicalWidth (49393 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, stateNone, selected,
        piDecNone, freshNone, c0None, c1None]
  | roundC0 coordinate =>
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      have selected := roundC0Grid_form? geometry coordinate
      rw [roundC0Target, roundC0MappedStart program] at selected
      change (roundC0Grid program).form? logicalWidth
        (15031534 + coordinate.val * 5328) =
          some ((RunningTransitionDirectPlan.Location.roundC0 coordinate).form
            geometry) at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [roundC0Target, roundC0MappedStart program]
      change (substitution program).form? logicalWidth
        (15031534 + coordinate.val * 5328) =
          some ((RunningTransitionDirectPlan.Location.roundC0 coordinate).form
            geometry)
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth
          (15031534 + coordinate.val * 5328) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth
          (15031534 + coordinate.val * 5328) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth
          (15031534 + coordinate.val * 5328) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth
          (15031534 + coordinate.val * 5328) (by omega)
      have c1None := roundC1Grid_form?_none_at_roundC0
        (program := program) (logicalWidth := logicalWidth) coordinate
      rw [roundC0Target, roundC0MappedStart program] at c1None
      change (roundC1Grid program).form? logicalWidth
        (15031534 + coordinate.val * 5328) = none at c1None
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, freshNone, selected, c1None]
  | roundC1 coordinate =>
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      have selected := roundC1Grid_form? geometry coordinate
      rw [roundC1Target, roundC1MappedStart program] at selected
      change (roundC1Grid program).form? logicalWidth
        (15032126 + coordinate.val * 5328) =
          some ((RunningTransitionDirectPlan.Location.roundC1 coordinate).form
            geometry) at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [roundC1Target, roundC1MappedStart program]
      change (substitution program).form? logicalWidth
        (15032126 + coordinate.val * 5328) =
          some ((RunningTransitionDirectPlan.Location.roundC1 coordinate).form
            geometry)
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth
          (15032126 + coordinate.val * 5328) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth
          (15032126 + coordinate.val * 5328) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_before
        (piDecRange program) logicalWidth
          (15032126 + coordinate.val * 5328) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth
          (15032126 + coordinate.val * 5328) (by omega)
      have c0None := roundC0Grid_form?_none_at_roundC1
        (program := program) (logicalWidth := logicalWidth) coordinate
      rw [roundC1Target, roundC1MappedStart program] at c0None
      change (roundC0Grid program).form? logicalWidth
        (15032126 + coordinate.val * 5328) = none at c0None
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, freshNone, c0None, selected]
  | piDec index =>
      have indexBound := index.isLt
      change index.val < 49248 at indexBound
      have selected := piDecRange_form? geometry index
      rw [piDecTarget, piDecMappedStart program] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [piDecTarget, piDecMappedStart program]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth (28972970 + index.val) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth (28972970 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (28972970 + index.val) (by omega)
      have c0None := SourceGrid.form?_eq_none_of_after
        (roundC0Grid program) logicalWidth (28972970 + index.val)
        (by rw [c0StrideValue]; omega)
        (by rw [c0StartValue, c0CountValue, c0StrideValue]; omega)
      have c1None := SourceGrid.form?_eq_none_of_after
        (roundC1Grid program) logicalWidth (28972970 + index.val)
        (by rw [c1StrideValue]; omega)
        (by rw [c1StartValue, c1CountValue, c1StrideValue]; omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        selected, freshNone, c0None, c1None]
  | fresh index =>
      have indexBound := index.isLt
      change index.val < 296138 at indexBound
      have selected := freshRange_form? geometry index
      rw [freshTarget, freshMappedStart program] at selected
      simp only [RunningTransitionDirectPlan.Location.sourceColumn]
      rw [freshTarget, freshMappedStart program]
      have stateNone := SourceRange.form?_eq_none_of_after
        (stateRange program) logicalWidth (29040308 + index.val) (by omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange program) logicalWidth (29040308 + index.val) (by omega)
      have piDecNone := SourceRange.form?_eq_none_of_after
        (piDecRange program) logicalWidth (29040308 + index.val) (by omega)
      have c0None := SourceGrid.form?_eq_none_of_after
        (roundC0Grid program) logicalWidth (29040308 + index.val)
        (by rw [c0StrideValue]; omega)
        (by rw [c0StartValue, c0CountValue, c0StrideValue]; omega)
      have c1None := SourceGrid.form?_eq_none_of_after
        (roundC1Grid program) logicalWidth (29040308 + index.val)
        (by rw [c1StrideValue]; omega)
        (by rw [c1StartValue, c1CountValue, c1StrideValue]; omega)
      simp [substitution, SourceSubstitution.form?, stateNone, outputNone,
        piDecNone, selected, c0None, c1None]

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
