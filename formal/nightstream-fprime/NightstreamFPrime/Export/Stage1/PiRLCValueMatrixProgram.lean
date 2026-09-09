import NightstreamFPrime.Export.Stage1.PiRLCValueWiring

/-!
Owns the compact invocation-key substitution for PiRLC product values. Six
structural mappings select the existing pilot and PiCCS retained forms.

This module does not allocate value copies or construct product rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCValueMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open PiCCSOrdinaryRetainedBlocks
open PiCCSOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def freshCommitmentRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (proofLogicalBlock program)
    (proofLogicalStart program) 0 1188 0

def runningCommitmentGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (priorInputBlock program) (priorInputStart program)
    1188 16 1188 1 1188 1188
    (PiCCSInputs.runningCommitmentStart 0 - PilotProduction.priorPreimageStart)
    PiCCSInputs.runningGroupWords 0

def freshPublicRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (freshPublicInputBlock program)
    (freshPublicInputStart program) 20196 270 0

def runningPublicGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (priorInputBlock program) (priorInputStart program)
    20466 16 270 1 270 270
    (PiCCSInputs.runningPublicStart 0 - PilotProduction.priorPreimageStart)
    PiCCSInputs.runningGroupWords 0

def evalKGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    24786 17 108 1 108 108
    (PiCCSInputs.outputEvaluationStart - PiCCSInputs.proofInputStart) 1620 0

def evalAGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    26622 17 1512 1 1512 1512
    (PiCCSInputs.outputEvaluationStart + 108 - PiCCSInputs.proofInputStart)
    1620 0

/-- Exact invocation-major value map. The six key domains are disjoint and
cover the four product families in their canonical order. -/
def substitution (program : Program) : SourceSubstitution where
  ranges := [freshCommitmentRange program, freshPublicRange program]
  grids := [runningCommitmentGrid program, runningPublicGrid program,
    evalKGrid program, evalAGrid program]

private theorem form_eq_location
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor)
    (location : PiCCSOrdinaryDirectPlan.Location)
    (owns : location.sourceColumn =
      descriptor.valueColumn descriptor.lane) :
    location.form geometry =
      PiRLCValueWiring.form geometry descriptor.invocation := by
  have left := PiCCSOrdinaryMatrixProgram.substitution_location_form?
    geometry location
  have right := PiCCSOrdinaryMatrixProgram.substitution_location_form?
    geometry (PiRLCValueWiring.located descriptor.invocation).location
  rw [owns] at left
  have right' :
      (PiCCSOrdinaryMatrixProgram.substitution program).form? logicalWidth
          (Spartan.sourceToSpartan
            (descriptor.valueColumn descriptor.lane)) =
        some ((PiRLCValueWiring.located descriptor.invocation).location.form
          geometry) := by
    simpa only [PiRLCValueWiring.location_sourceColumn,
      PiRLCProductSchedule.descriptor_invocation] using right
  rw [PiRLCValueWiring.form]
  exact Option.some.inj (left.symm.trans right')

private theorem freshCommitmentRange_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin 1188) :
    (substitution program).form? logicalWidth (0 + index.val) =
      some ((proofLogicalBlock program).form (proofLogicalStart program)
        (proofLogicalFits geometry) ⟨0 + index.val, by
          change 0 + index.val < 114878
          have bound := index.isLt
          omega⟩) := by
  have bound := index.isLt
  have selected := SourceRange.form?_ofSemantic (proofLogicalBlock program)
    (proofLogicalStart program) 0 1188 0 (proofLogicalFits geometry)
    (by change 0 + 1188 ≤ 114878; decide) index
  change (freshCommitmentRange program).form? logicalWidth (0 + index.val) = _ at selected
  have none_freshPublicRange := SourceRange.form?_eq_none_of_before
    (freshPublicRange program) logicalWidth (0 + index.val) (by
      change 0 + index.val < 20196; omega)
  have none_runningCommitmentGrid := SourceGrid.form?_eq_none_of_before
    (runningCommitmentGrid program) logicalWidth (0 + index.val) (by
      change 0 + index.val < 1188; omega)
  have none_runningPublicGrid := SourceGrid.form?_eq_none_of_before
    (runningPublicGrid program) logicalWidth (0 + index.val) (by
      change 0 + index.val < 20466; omega)
  have none_evalKGrid := SourceGrid.form?_eq_none_of_before
    (evalKGrid program) logicalWidth (0 + index.val) (by
      change 0 + index.val < 24786; omega)
  have none_evalAGrid := SourceGrid.form?_eq_none_of_before
    (evalAGrid program) logicalWidth (0 + index.val) (by
      change 0 + index.val < 26622; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshPublicRange,
    none_runningCommitmentGrid,
    none_runningPublicGrid,
    none_evalKGrid,
    none_evalAGrid]

private theorem runningCommitmentGrid_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin 16)
    (index : Fin 1188) :
    (substitution program).form? logicalWidth (1188 + source.val * 1188 + index.val) =
      some ((priorInputBlock program).form (priorInputStart program)
        (priorInputFits geometry) ⟨97 + source.val * 3081 + index.val, by
          change 97 + source.val * 3081 + index.val < 49393
          have bound := index.isLt
          have sources := source.isLt
          omega⟩) := by
  have bound := index.isLt
  have sources := source.isLt
  have selected := SourceGrid.form?_ofSemantic (priorInputBlock program)
    (priorInputStart program) 1188 16 1188 1 1188 1188
    97 3081 0 (priorInputFits geometry) (by decide) (by decide)
    source ⟨0, by decide⟩ index (by simpa using index.isLt)
    index.isLt (by change 97 + source.val * 3081 + index.val < 49393; omega)
  change (runningCommitmentGrid program).form? logicalWidth
    (1188 + source.val * 1188 + 0 * 1188 + index.val) = _ at selected
  simp only [Nat.zero_mul, Nat.add_zero] at selected
  have none_freshCommitmentRange := SourceRange.form?_eq_none_of_after
    (freshCommitmentRange program) logicalWidth (1188 + source.val * 1188 + index.val) (by
      change 0 + 1188 ≤ 1188 + source.val * 1188 + index.val; omega)
  have none_freshPublicRange := SourceRange.form?_eq_none_of_before
    (freshPublicRange program) logicalWidth (1188 + source.val * 1188 + index.val) (by
      change 1188 + source.val * 1188 + index.val < 20196; omega)
  have none_runningPublicGrid := SourceGrid.form?_eq_none_of_before
    (runningPublicGrid program) logicalWidth (1188 + source.val * 1188 + index.val) (by
      change 1188 + source.val * 1188 + index.val < 20466; omega)
  have none_evalKGrid := SourceGrid.form?_eq_none_of_before
    (evalKGrid program) logicalWidth (1188 + source.val * 1188 + index.val) (by
      change 1188 + source.val * 1188 + index.val < 24786; omega)
  have none_evalAGrid := SourceGrid.form?_eq_none_of_before
    (evalAGrid program) logicalWidth (1188 + source.val * 1188 + index.val) (by
      change 1188 + source.val * 1188 + index.val < 26622; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshCommitmentRange,
    none_freshPublicRange,
    none_runningPublicGrid,
    none_evalKGrid,
    none_evalAGrid]

private theorem freshPublicRange_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin 270) :
    (substitution program).form? logicalWidth (20196 + index.val) =
      some ((freshPublicInputBlock program).form (freshPublicInputStart program)
        (freshPublicInputFits geometry) ⟨0 + index.val, by
          change 0 + index.val < 270
          have bound := index.isLt
          omega⟩) := by
  have bound := index.isLt
  have selected := SourceRange.form?_ofSemantic (freshPublicInputBlock program)
    (freshPublicInputStart program) 20196 270 0 (freshPublicInputFits geometry)
    (by change 0 + 270 ≤ 270; decide) index
  change (freshPublicRange program).form? logicalWidth (20196 + index.val) = _ at selected
  have none_freshCommitmentRange := SourceRange.form?_eq_none_of_after
    (freshCommitmentRange program) logicalWidth (20196 + index.val) (by
      change 0 + 1188 ≤ 20196 + index.val; omega)
  have none_runningCommitmentGrid := SourceGrid.form?_eq_none_of_after
    (runningCommitmentGrid program) logicalWidth (20196 + index.val) (by change 0 < 1188; decide) (by
      change 1188 + 16 * 1188 ≤ 20196 + index.val; omega)
  have none_runningPublicGrid := SourceGrid.form?_eq_none_of_before
    (runningPublicGrid program) logicalWidth (20196 + index.val) (by
      change 20196 + index.val < 20466; omega)
  have none_evalKGrid := SourceGrid.form?_eq_none_of_before
    (evalKGrid program) logicalWidth (20196 + index.val) (by
      change 20196 + index.val < 24786; omega)
  have none_evalAGrid := SourceGrid.form?_eq_none_of_before
    (evalAGrid program) logicalWidth (20196 + index.val) (by
      change 20196 + index.val < 26622; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshCommitmentRange,
    none_runningCommitmentGrid,
    none_runningPublicGrid,
    none_evalKGrid,
    none_evalAGrid]

private theorem runningPublicGrid_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin 16)
    (index : Fin 270) :
    (substitution program).form? logicalWidth (20466 + source.val * 270 + index.val) =
      some ((priorInputBlock program).form (priorInputStart program)
        (priorInputFits geometry) ⟨1286 + source.val * 3081 + index.val, by
          change 1286 + source.val * 3081 + index.val < 49393
          have bound := index.isLt
          have sources := source.isLt
          omega⟩) := by
  have bound := index.isLt
  have sources := source.isLt
  have selected := SourceGrid.form?_ofSemantic (priorInputBlock program)
    (priorInputStart program) 20466 16 270 1 270 270
    1286 3081 0 (priorInputFits geometry) (by decide) (by decide)
    source ⟨0, by decide⟩ index (by simpa using index.isLt)
    index.isLt (by change 1286 + source.val * 3081 + index.val < 49393; omega)
  change (runningPublicGrid program).form? logicalWidth
    (20466 + source.val * 270 + 0 * 270 + index.val) = _ at selected
  simp only [Nat.zero_mul, Nat.add_zero] at selected
  have none_freshCommitmentRange := SourceRange.form?_eq_none_of_after
    (freshCommitmentRange program) logicalWidth (20466 + source.val * 270 + index.val) (by
      change 0 + 1188 ≤ 20466 + source.val * 270 + index.val; omega)
  have none_freshPublicRange := SourceRange.form?_eq_none_of_after
    (freshPublicRange program) logicalWidth (20466 + source.val * 270 + index.val) (by
      change 20196 + 270 ≤ 20466 + source.val * 270 + index.val; omega)
  have none_runningCommitmentGrid := SourceGrid.form?_eq_none_of_after
    (runningCommitmentGrid program) logicalWidth (20466 + source.val * 270 + index.val) (by change 0 < 1188; decide) (by
      change 1188 + 16 * 1188 ≤ 20466 + source.val * 270 + index.val; omega)
  have none_evalKGrid := SourceGrid.form?_eq_none_of_before
    (evalKGrid program) logicalWidth (20466 + source.val * 270 + index.val) (by
      change 20466 + source.val * 270 + index.val < 24786; omega)
  have none_evalAGrid := SourceGrid.form?_eq_none_of_before
    (evalAGrid program) logicalWidth (20466 + source.val * 270 + index.val) (by
      change 20466 + source.val * 270 + index.val < 26622; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshCommitmentRange,
    none_freshPublicRange,
    none_runningCommitmentGrid,
    none_evalKGrid,
    none_evalAGrid]

private theorem evalKGrid_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin 17)
    (index : Fin 108) :
    (substitution program).form? logicalWidth (24786 + source.val * 108 + index.val) =
      some ((proofLogicalBlock program).form (proofLogicalStart program)
        (proofLogicalFits geometry) ⟨1748 + source.val * 1620 + index.val, by
          change 1748 + source.val * 1620 + index.val < 114878
          have bound := index.isLt
          have sources := source.isLt
          omega⟩) := by
  have bound := index.isLt
  have sources := source.isLt
  have selected := SourceGrid.form?_ofSemantic (proofLogicalBlock program)
    (proofLogicalStart program) 24786 17 108 1 108 108
    1748 1620 0 (proofLogicalFits geometry) (by decide) (by decide)
    source ⟨0, by decide⟩ index (by simpa using index.isLt)
    index.isLt (by change 1748 + source.val * 1620 + index.val < 114878; omega)
  change (evalKGrid program).form? logicalWidth
    (24786 + source.val * 108 + 0 * 108 + index.val) = _ at selected
  simp only [Nat.zero_mul, Nat.add_zero] at selected
  have none_freshCommitmentRange := SourceRange.form?_eq_none_of_after
    (freshCommitmentRange program) logicalWidth (24786 + source.val * 108 + index.val) (by
      change 0 + 1188 ≤ 24786 + source.val * 108 + index.val; omega)
  have none_freshPublicRange := SourceRange.form?_eq_none_of_after
    (freshPublicRange program) logicalWidth (24786 + source.val * 108 + index.val) (by
      change 20196 + 270 ≤ 24786 + source.val * 108 + index.val; omega)
  have none_runningCommitmentGrid := SourceGrid.form?_eq_none_of_after
    (runningCommitmentGrid program) logicalWidth (24786 + source.val * 108 + index.val) (by change 0 < 1188; decide) (by
      change 1188 + 16 * 1188 ≤ 24786 + source.val * 108 + index.val; omega)
  have none_runningPublicGrid := SourceGrid.form?_eq_none_of_after
    (runningPublicGrid program) logicalWidth (24786 + source.val * 108 + index.val) (by change 0 < 270; decide) (by
      change 20466 + 16 * 270 ≤ 24786 + source.val * 108 + index.val; omega)
  have none_evalAGrid := SourceGrid.form?_eq_none_of_before
    (evalAGrid program) logicalWidth (24786 + source.val * 108 + index.val) (by
      change 24786 + source.val * 108 + index.val < 26622; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshCommitmentRange,
    none_freshPublicRange,
    none_runningCommitmentGrid,
    none_runningPublicGrid,
    none_evalAGrid]

private theorem evalAGrid_lookup {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin 17)
    (index : Fin 1512) :
    (substitution program).form? logicalWidth (26622 + source.val * 1512 + index.val) =
      some ((proofLogicalBlock program).form (proofLogicalStart program)
        (proofLogicalFits geometry) ⟨1856 + source.val * 1620 + index.val, by
          change 1856 + source.val * 1620 + index.val < 114878
          have bound := index.isLt
          have sources := source.isLt
          omega⟩) := by
  have bound := index.isLt
  have sources := source.isLt
  have selected := SourceGrid.form?_ofSemantic (proofLogicalBlock program)
    (proofLogicalStart program) 26622 17 1512 1 1512 1512
    1856 1620 0 (proofLogicalFits geometry) (by decide) (by decide)
    source ⟨0, by decide⟩ index (by simpa using index.isLt)
    index.isLt (by change 1856 + source.val * 1620 + index.val < 114878; omega)
  change (evalAGrid program).form? logicalWidth
    (26622 + source.val * 1512 + 0 * 1512 + index.val) = _ at selected
  simp only [Nat.zero_mul, Nat.add_zero] at selected
  have none_freshCommitmentRange := SourceRange.form?_eq_none_of_after
    (freshCommitmentRange program) logicalWidth (26622 + source.val * 1512 + index.val) (by
      change 0 + 1188 ≤ 26622 + source.val * 1512 + index.val; omega)
  have none_freshPublicRange := SourceRange.form?_eq_none_of_after
    (freshPublicRange program) logicalWidth (26622 + source.val * 1512 + index.val) (by
      change 20196 + 270 ≤ 26622 + source.val * 1512 + index.val; omega)
  have none_runningCommitmentGrid := SourceGrid.form?_eq_none_of_after
    (runningCommitmentGrid program) logicalWidth (26622 + source.val * 1512 + index.val) (by change 0 < 1188; decide) (by
      change 1188 + 16 * 1188 ≤ 26622 + source.val * 1512 + index.val; omega)
  have none_runningPublicGrid := SourceGrid.form?_eq_none_of_after
    (runningPublicGrid program) logicalWidth (26622 + source.val * 1512 + index.val) (by change 0 < 270; decide) (by
      change 20466 + 16 * 270 ≤ 26622 + source.val * 1512 + index.val; omega)
  have none_evalKGrid := SourceGrid.form?_eq_none_of_after
    (evalKGrid program) logicalWidth (26622 + source.val * 1512 + index.val) (by change 0 < 108; decide) (by
      change 24786 + 17 * 108 ≤ 26622 + source.val * 1512 + index.val; omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
    List.filterMap_nil, List.nil_append, List.append_nil, selected,
    none_freshCommitmentRange,
    none_freshPublicRange,
    none_runningCommitmentGrid,
    none_runningPublicGrid,
    none_evalKGrid]

private theorem substitution_form?_descriptor
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    (substitution program).form? logicalWidth descriptor.invocation.val =
      some (PiRLCValueWiring.form geometry descriptor.invocation) := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family with
  | commitment =>
      by_cases first : source.val = 0
      · have sourceZero : source = ⟨0, by
            norm_num [PiRLCCombinationInvocations.sourceCount]⟩ := by
          apply Fin.ext
          exact first
        subst source
        let index : Fin PiCCSOrdinaryRetainedBlocks.proofInputCount :=
          ⟨block.val * ringDegree + lane.val, by
            have blockBound := block.isLt
            have laneBound := lane.isLt
            rw [PiCCSOrdinaryRetainedBlocks.proofInputCount_eq]
            norm_num [PiRLCProductSchedule.Family.blockCount, ringDegree]
              at blockBound laneBound ⊢
            omega⟩
        let location : PiCCSOrdinaryDirectPlan.Location :=
          .proofLogical (PiCCSOrdinaryRetainedBlocks.proofInputSlot index)
        let zeroSource : Fin PiRLCCombinationInvocations.sourceCount :=
          ⟨0, by norm_num [PiRLCCombinationInvocations.sourceCount]⟩
        have owns : location.sourceColumn =
            (⟨.commitment, zeroSource, block, lane, cell⟩ :
              PiRLCProductSchedule.Descriptor).valueColumn lane := by
          simp [location, index, zeroSource,
            PiCCSOrdinaryDirectPlan.Location.sourceColumn,
            PiCCSOrdinaryRetainedBlocks.proofLogicalSource_proofInput,
            PiRLCProductSchedule.Descriptor.valueColumn,
            PiRLCCombinationInvocations.commitmentValueSourceStart,
            PiCCSInputs.freshCommitmentStart] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
        rw [← form_eq_location geometry _ location owns]
        let operand : Fin 1188 := ⟨block.val * 54 + lane.val, by
          have b := block.isLt
          have l := lane.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
          omega⟩
        have key : (⟨.commitment, zeroSource, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).invocation.val = 0 + operand.val := by
          have b := block.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount] at b c
          simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, operand, zeroSource]
          omega
        rw [key]
        change _ = some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
          (proofInputSlot index)).form geometry)
        rw [PiCCSOrdinaryDirectPlan.Location.form_proofInput]
        simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
          operand, index, proofInputSlot, ringDegree,
          Nat.zero_add, Nat.add_assoc] using freshCommitmentRange_lookup geometry operand
      · let index : Fin PilotProduction.stateHashWords :=
          ⟨97 + (source.val - 1) * 3081 + block.val * ringDegree + lane.val, by
            have sourceBound := source.isLt
            have blockBound := block.isLt
            have laneBound := lane.isLt
            norm_num [PiRLCCombinationInvocations.sourceCount,
              PiRLCProductSchedule.Family.blockCount,
              PilotProduction.stateHashWords_eq, ringDegree]
              at sourceBound blockBound laneBound ⊢
            omega⟩
        let location : PiCCSOrdinaryDirectPlan.Location := .priorInput index
        have owns : location.sourceColumn =
            (⟨.commitment, source, block, lane, cell⟩ :
              PiRLCProductSchedule.Descriptor).valueColumn lane := by
          simp [location, index, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCProductSchedule.Descriptor.valueColumn,
            PiRLCCombinationInvocations.commitmentValueSourceStart,
            PiCCSInputs.runningCommitmentStart, PiCCSInputs.runningGroupStart,
            PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
            PilotProduction.priorPreimageStart, PiCCSInputs.runningGroupWords,
            first] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
        rw [← form_eq_location geometry _ location owns]
        let operand : Fin 1188 := ⟨block.val * 54 + lane.val, by
          have b := block.isLt
          have l := lane.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
          omega⟩
        let priorSource : Fin 16 := ⟨source.val - 1, by
          have bound := source.isLt
          change source.val < 17 at bound
          omega⟩
        have key : (⟨.commitment, source, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).invocation.val = 1188 + priorSource.val * 1188 + operand.val := by
          have b := block.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount] at b c
          simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, operand, priorSource]
          omega
        rw [key]
        simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
          operand, index, proofInputSlot, priorSource, ringDegree,
          Nat.zero_add, Nat.add_assoc] using runningCommitmentGrid_lookup geometry priorSource operand
  | publicInput =>
      by_cases first : source.val = 0
      · have sourceZero : source = ⟨0, by
            norm_num [PiRLCCombinationInvocations.sourceCount]⟩ := by
          apply Fin.ext
          exact first
        subst source
        let index : Fin 270 := ⟨block.val * ringDegree + lane.val, by
          have blockBound := block.isLt
          have laneBound := lane.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount, ringDegree]
            at blockBound laneBound ⊢
          omega⟩
        let location : PiCCSOrdinaryDirectPlan.Location := .freshPublicInput index
        let zeroSource : Fin PiRLCCombinationInvocations.sourceCount :=
          ⟨0, by norm_num [PiRLCCombinationInvocations.sourceCount]⟩
        have owns : location.sourceColumn =
            (⟨.publicInput, zeroSource, block, lane, cell⟩ :
              PiRLCProductSchedule.Descriptor).valueColumn lane := by
          simp [location, index, zeroSource,
            PiCCSOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCProductSchedule.Descriptor.valueColumn,
            PiRLCCombinationInvocations.publicInputValueSourceStart] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
        rw [← form_eq_location geometry _ location owns]
        let operand : Fin 270 := ⟨block.val * 54 + lane.val, by
          have b := block.isLt
          have l := lane.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
          omega⟩
        have key : (⟨.publicInput, zeroSource, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).invocation.val = 20196 + operand.val := by
          have b := block.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount] at b c
          simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, operand, zeroSource]
          omega
        rw [key]
        simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
          operand, index, proofInputSlot, ringDegree,
          Nat.zero_add, Nat.add_assoc] using freshPublicRange_lookup geometry operand
      · let index : Fin PilotProduction.stateHashWords :=
          ⟨1286 + (source.val - 1) * 3081 + block.val * ringDegree + lane.val, by
            have sourceBound := source.isLt
            have blockBound := block.isLt
            have laneBound := lane.isLt
            norm_num [PiRLCCombinationInvocations.sourceCount,
              PiRLCProductSchedule.Family.blockCount,
              PilotProduction.stateHashWords_eq, ringDegree]
              at sourceBound blockBound laneBound ⊢
            omega⟩
        let location : PiCCSOrdinaryDirectPlan.Location := .priorInput index
        have owns : location.sourceColumn =
            (⟨.publicInput, source, block, lane, cell⟩ :
              PiRLCProductSchedule.Descriptor).valueColumn lane := by
          simp [location, index, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCProductSchedule.Descriptor.valueColumn,
            PiRLCCombinationInvocations.publicInputValueSourceStart,
            PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
            PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
            PilotProduction.priorPreimageStart, PiCCSInputs.runningGroupWords,
            first] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
        rw [← form_eq_location geometry _ location owns]
        let operand : Fin 270 := ⟨block.val * 54 + lane.val, by
          have b := block.isLt
          have l := lane.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
          omega⟩
        let priorSource : Fin 16 := ⟨source.val - 1, by
          have bound := source.isLt
          change source.val < 17 at bound
          omega⟩
        have key : (⟨.publicInput, source, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).invocation.val = 20466 + priorSource.val * 270 + operand.val := by
          have b := block.isLt
          have c := cell.isLt
          norm_num [PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount] at b c
          simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, operand, priorSource]
          omega
        rw [key]
        simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
          operand, index, proofInputSlot, priorSource, ringDegree,
          Nat.zero_add, Nat.add_assoc] using runningPublicGrid_lookup geometry priorSource operand
  | evalK =>
      let index : Fin PiCCSOrdinaryRetainedBlocks.proofInputCount :=
        ⟨1748 + source.val * 1620 + lane.val * 2 + cell.val, by
          have sourceBound := source.isLt
          have laneBound := lane.isLt
          have cellBound := cell.isLt
          rw [PiCCSOrdinaryRetainedBlocks.proofInputCount_eq]
          norm_num [PiRLCCombinationInvocations.sourceCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree]
            at sourceBound laneBound cellBound ⊢
          omega⟩
      let location : PiCCSOrdinaryDirectPlan.Location :=
        .proofLogical (PiCCSOrdinaryRetainedBlocks.proofInputSlot index)
      have owns : location.sourceColumn =
          (⟨.evalK, source, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).valueColumn lane := by
        simp [location, index, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
          PiCCSOrdinaryRetainedBlocks.proofLogicalSource_proofInput,
          PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.evalKValueSourceStart,
          PiCCSInputs.outputEvaluationStart, PiCCSInputs.roundMessageStart,
          PiCCSInputs.freshCommitmentStart] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
      rw [← form_eq_location geometry _ location owns]
      let operand : Fin 108 := ⟨lane.val * 2 + cell.val, by
        have b := block.isLt
        have l := lane.isLt
        have c := cell.isLt
        norm_num [PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
        omega⟩
      have key : (⟨.evalK, source, block, lane, cell⟩ :
          PiRLCProductSchedule.Descriptor).invocation.val = 24786 + source.val * 108 + operand.val := by
        have b := block.isLt
        have c := cell.isLt
        norm_num [PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount] at b c
        simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount, operand]
        omega
      rw [key]
      change _ = some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputSlot index)).form geometry)
      rw [PiCCSOrdinaryDirectPlan.Location.form_proofInput]
      simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
        operand, index, proofInputSlot, ringDegree,
        Nat.zero_add, Nat.add_assoc] using evalKGrid_lookup geometry source operand
  | evalA =>
      let index : Fin PiCCSOrdinaryRetainedBlocks.proofInputCount :=
        ⟨1856 + source.val * 1620 + block.val * 108 + lane.val * 2 + cell.val, by
          have sourceBound := source.isLt
          have blockBound := block.isLt
          have laneBound := lane.isLt
          have cellBound := cell.isLt
          rw [PiCCSOrdinaryRetainedBlocks.proofInputCount_eq]
          norm_num [PiRLCCombinationInvocations.sourceCount,
            PiRLCProductSchedule.Family.blockCount,
            PiRLCProductSchedule.Family.cellCount, ringDegree]
            at sourceBound blockBound laneBound cellBound ⊢
          omega⟩
      let location : PiCCSOrdinaryDirectPlan.Location :=
        .proofLogical (PiCCSOrdinaryRetainedBlocks.proofInputSlot index)
      have owns : location.sourceColumn =
          (⟨.evalA, source, block, lane, cell⟩ :
            PiRLCProductSchedule.Descriptor).valueColumn lane := by
        simp [location, index, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
          PiCCSOrdinaryRetainedBlocks.proofLogicalSource_proofInput,
          PiRLCProductSchedule.Descriptor.valueColumn,
          PiRLCCombinationInvocations.evalAValueSourceStart,
          PiCCSInputs.outputEvaluationStart, PiCCSInputs.roundMessageStart,
          PiCCSInputs.freshCommitmentStart] <;> norm_num [PiCCSInputs.freshCommitmentWords, PiCCSInputs.roundMessageWords] <;> omega
      rw [← form_eq_location geometry _ location owns]
      let operand : Fin 1512 := ⟨block.val * 108 + lane.val * 2 + cell.val, by
        have b := block.isLt
        have l := lane.isLt
        have c := cell.isLt
        norm_num [PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount, ringDegree] at b l c
        omega⟩
      have key : (⟨.evalA, source, block, lane, cell⟩ :
          PiRLCProductSchedule.Descriptor).invocation.val = 26622 + source.val * 1512 + operand.val := by
        have b := block.isLt
        have c := cell.isLt
        norm_num [PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount] at b c
        simp only [PiRLCProductSchedule.Descriptor.invocation_val, PiRLCProductSchedule.Family.blockCount,
          PiRLCProductSchedule.Family.cellCount, operand]
        omega
      rw [key]
      change _ = some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputSlot index)).form geometry)
      rw [PiCCSOrdinaryDirectPlan.Location.form_proofInput]
      simpa only [location, PiCCSOrdinaryDirectPlan.Location.form,
        operand, index, proofInputSlot, ringDegree,
        Nat.zero_add, Nat.add_assoc] using evalAGrid_lookup geometry source operand

/-- Every compact input key resolves to its existing authoritative PiCCS
form. The proof is symbolic in the descriptor and does not expand the table. -/
theorem substitution_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (invocation : Fin PiRLCProductSchedule.invocationCount) :
    (substitution program).form? logicalWidth invocation.val =
      some (PiRLCValueWiring.form geometry invocation) := by
  have selected := substitution_form?_descriptor geometry
    (PiRLCProductSchedule.descriptor invocation)
  simpa only [PiRLCProductSchedule.invocation_descriptor] using selected

end NightstreamFPrime.Export.Stage1.PiRLCValueMatrixProgram
