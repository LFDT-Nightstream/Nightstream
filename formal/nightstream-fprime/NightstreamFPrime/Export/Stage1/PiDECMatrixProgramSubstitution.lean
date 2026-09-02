import NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

/-!
Proves exact source custody for the compact PiDEC ordinary-row program. Every
source column used by a canonical PiDEC row resolves through exactly one
Lean-authored retained range and equals the proof-oriented direct source map.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open PiDECRetainedBlocks
open PiDECRetainedGeometry

private theorem rangeValues (program : ApplicationProgram) :
    (parentCommitmentRange program).sourceStart = 20347121 ∧
    (parentCommitmentRange program).sourceCount = 1188 ∧
    (parentPublicInputRange program).sourceStart = 20352629 ∧
    (parentPublicInputRange program).sourceCount = 270 ∧
    (parentEvalKRange program).sourceStart = 20354627 ∧
    (parentEvalKRange program).sourceCount = 108 ∧
    (parentEvalARange program).sourceStart = 20378927 ∧
    (parentEvalARange program).sourceCount = 1512 ∧
    (proofRange program).sourceStart = 28972970 ∧
    (proofRange program).sourceCount = 49248 ∧
    (logicalRange program).sourceStart = 29022218 ∧
    (logicalRange program).sourceCount = 270 ∧
    (freshRange program).sourceStart = 29022488 ∧
    (freshRange program).sourceCount = 17820 := by
  norm_num [parentCommitmentRange, parentPublicInputRange, parentEvalKRange,
    parentEvalARange, proofRange, logicalRange, freshRange,
    SourceRange.ofSemantic, Spartan.sourceToSpartan,
    PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
    PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
    PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
    PiDECInputs.publicInputWordsPerChild, PiDECInputs.phaseOffset,
    PiDECStarts.phaseLogicalStart,
    PiDECStarts.phaseFreshStart, PiDECSourceSupport.freshCount,
    Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount,
    Spartan.pilotSourceColumnCount,
    Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
    Spartan.piCcsLocalStart, Spartan.pilotInputPrivateColumnCount,
    Spartan.expectedContextPublicStart]

private theorem parentCommitmentTarget (program : ApplicationProgram)
    (index : Fin PiDECInputs.commitmentWordsPerChild) :
    Spartan.sourceToSpartan
        (PiDECSourceSupport.parentCommitmentStart + index.val) =
      20347121 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · rw [PiDECSourceSupport.parentCommitmentStart_eq]
    norm_num [Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · rw [PiDECSourceSupport.parentCommitmentStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]

private theorem parentPublicInputTarget (program : ApplicationProgram)
    (index : Fin PiDECInputs.publicInputWordsPerChild) :
    Spartan.sourceToSpartan
        (PiDECSourceSupport.parentPublicInputStart + index.val) =
      20352629 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · rw [PiDECSourceSupport.parentPublicInputStart_eq]
    norm_num [Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · rw [PiDECSourceSupport.parentPublicInputStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]

private theorem parentEvalKTarget (program : ApplicationProgram)
    (index : Fin PiDECInputs.evalKWordsPerChild) :
    Spartan.sourceToSpartan
        (PiDECSourceSupport.parentEvalKStart + index.val) =
      20354627 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · rw [PiDECSourceSupport.parentEvalKStart_eq]
    norm_num [Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · rw [PiDECSourceSupport.parentEvalKStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]

private theorem parentEvalATarget (program : ApplicationProgram)
    (index : Fin PiDECInputs.evalAWordsPerChild) :
    Spartan.sourceToSpartan
        (PiDECSourceSupport.parentEvalAStart + index.val) =
      20378927 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · rw [PiDECSourceSupport.parentEvalAStart_eq]
    norm_num [Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · rw [PiDECSourceSupport.parentEvalAStart_eq]
    norm_num [Spartan.piCcsPhaseOffset]

private theorem proofTarget (program : ApplicationProgram)
    (index : Fin PiDECInputs.proofInputColumnCount) :
    Spartan.sourceToSpartan (PiDECInputs.proofInputStart + index.val) =
      28972970 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · norm_num [PiDECInputs.proofInputStart, Spartan.sourceToSpartan,
      Spartan.pilotSourceColumnCount, Spartan.proofInputSourceStart,
      Spartan.piCcsPhaseOffset, Spartan.piCcsLocalStart]
  · norm_num [PiDECInputs.proofInputStart, Spartan.piCcsPhaseOffset]

private theorem logicalTarget (program : ApplicationProgram)
    (index : Fin 270) :
    Spartan.sourceToSpartan (PiDECStarts.phaseLogicalStart + index.val) =
      29022218 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild, Spartan.sourceToSpartan,
      Spartan.pilotSourceColumnCount, Spartan.proofInputSourceStart,
      Spartan.piCcsPhaseOffset, Spartan.piCcsLocalStart]
  · norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild, Spartan.piCcsPhaseOffset]

private theorem freshTarget (program : ApplicationProgram)
    (index : Fin freshCount) :
    Spartan.sourceToSpartan (PiDECStarts.phaseFreshStart + index.val) =
      29022488 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal]
  · norm_num [PiDECStarts.phaseFreshStart, PiDECStarts.phaseLogicalStart,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
      Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount,
      Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · norm_num [PiDECStarts.phaseFreshStart, PiDECStarts.phaseLogicalStart,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
      Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount,
      Spartan.piCcsPhaseOffset]

theorem parentCommitmentRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.commitmentWordsPerChild) :
    (parentCommitmentRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentCommitmentStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentCommitment index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECSourceSupport.parentCommitmentStart index.val (by
      rw [PiDECSourceSupport.parentCommitmentStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  simpa [parentCommitmentRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentCommitmentBlock program)
      (PiDECRetainedGeometry.parentCommitmentStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart)
      PiDECInputs.commitmentWordsPerChild 0
      (parentCommitmentFits geometry) (by rfl) index)

theorem parentPublicInputRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.publicInputWordsPerChild) :
    (parentPublicInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentPublicInputStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentPublicInput index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECSourceSupport.parentPublicInputStart index.val (by
      rw [PiDECSourceSupport.parentPublicInputStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  simpa [parentPublicInputRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentPublicInputBlock program)
      (PiDECRetainedGeometry.parentPublicInputStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart)
      PiDECInputs.publicInputWordsPerChild 0
      (parentPublicInputFits geometry) (by rfl) index)

theorem parentEvalKRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.evalKWordsPerChild) :
    (parentEvalKRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentEvalKStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentEvalK index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECSourceSupport.parentEvalKStart index.val (by
      rw [PiDECSourceSupport.parentEvalKStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  simpa [parentEvalKRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentEvalKBlock program)
      (PiDECRetainedGeometry.parentEvalKStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart)
      PiDECInputs.evalKWordsPerChild 0
      (parentEvalKFits geometry) (by rfl) index)

theorem parentEvalARange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.evalAWordsPerChild) :
    (parentEvalARange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentEvalAStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentEvalA index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECSourceSupport.parentEvalAStart index.val (by
      rw [PiDECSourceSupport.parentEvalAStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  simpa [parentEvalARange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentEvalABlock program)
      (PiDECRetainedGeometry.parentEvalAStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart)
      PiDECInputs.evalAWordsPerChild 0
      (parentEvalAFits geometry) (by rfl) index)

theorem proofRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.proofInputColumnCount) :
    (proofRange program).form? logicalWidth
        (Spartan.sourceToSpartan (PiDECInputs.proofInputStart + index.val)) =
      some ((PiDECDirectPlan.Location.proof index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECInputs.proofInputStart index.val (by
      norm_num [PiDECInputs.proofInputStart, Spartan.piCcsPhaseOffset])]
  simpa [proofRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (proofBlock program) (proofStart program)
      (Spartan.sourceToSpartan PiDECInputs.proofInputStart)
      PiDECInputs.proofInputColumnCount 0 (proofFits geometry) (by rfl) index)

theorem logicalRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (logicalRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECStarts.phaseLogicalStart + index.val)) =
      some ((PiDECDirectPlan.Location.logical index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECStarts.phaseLogicalStart index.val (by
      norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
        PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
        PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
        PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild, Spartan.piCcsPhaseOffset])]
  simpa [logicalRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (logicalBlock program) (logicalStart program)
      (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart) 270 0
      (logicalFits geometry) (by rfl) index)

theorem freshRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin freshCount) :
    (freshRange program).form? logicalWidth
        (Spartan.sourceToSpartan (PiDECStarts.phaseFreshStart + index.val)) =
      some ((PiDECDirectPlan.Location.fresh index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiDECStarts.phaseFreshStart index.val (by
      norm_num [PiDECStarts.phaseFreshStart, PiDECStarts.phaseLogicalStart,
        PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
        Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount,
        Spartan.piCcsPhaseOffset])]
  simpa [freshRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshBlock program) (freshStart program)
      (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart) freshCount 0
      (freshFits geometry) (by rfl) index)

/-- The compact substitution reconstructs every direct PiDEC source
location and rejects all overlapping interpretations. -/
theorem substitution_location_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PiDECDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  rcases rangeValues program with
    ⟨commitStart, commitCount, publicStart, publicCount, evalKStart,
      evalKCount, evalAStart, evalACount, proofStartValue, proofCount,
      logicalStartValue, logicalCount, freshStartValue, freshCountValue⟩
  cases location with
  | parentCommitment index =>
      have indexBound := index.isLt
      norm_num [PiDECInputs.commitmentWordsPerChild] at indexBound
      have target := parentCommitmentTarget program index
      have selected := parentCommitmentRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have publicNone := SourceRange.form?_eq_none_of_before
        (parentPublicInputRange program) logicalWidth (20347121 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_before
        (parentEvalKRange program) logicalWidth (20347121 + index.val) (by omega)
      have evalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (20347121 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (20347121 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (20347121 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (20347121 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, selected, publicNone,
        evalKNone, evalANone, proofNone, logicalNone, freshNone]
  | parentPublicInput index =>
      have indexBound := index.isLt
      norm_num [PiDECInputs.publicInputWordsPerChild] at indexBound
      have target := parentPublicInputTarget program index
      have selected := parentPublicInputRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (20352629 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_before
        (parentEvalKRange program) logicalWidth (20352629 + index.val) (by omega)
      have evalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (20352629 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (20352629 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (20352629 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (20352629 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, selected,
        evalKNone, evalANone, proofNone, logicalNone, freshNone]
  | parentEvalK index =>
      have indexBound := index.isLt
      norm_num [PiDECInputs.evalKWordsPerChild] at indexBound
      have target := parentEvalKTarget program index
      have selected := parentEvalKRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (20354627 + index.val)
        (by omega)
      have publicNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (20354627 + index.val)
        (by omega)
      have evalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (20354627 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (20354627 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (20354627 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (20354627 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, publicNone,
        selected, evalANone, proofNone, logicalNone, freshNone]
  | parentEvalA index =>
      have indexBound := index.isLt
      norm_num [PiDECInputs.evalAWordsPerChild] at indexBound
      have target := parentEvalATarget program index
      have selected := parentEvalARange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (20378927 + index.val)
        (by omega)
      have publicNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (20378927 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (20378927 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (20378927 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (20378927 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (20378927 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, publicNone,
        evalKNone, selected, proofNone, logicalNone, freshNone]
  | proof index =>
      have indexBound := index.isLt
      norm_num [PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild] at indexBound
      have target := proofTarget program index
      have selected := proofRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (28972970 + index.val)
        (by omega)
      have publicNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (28972970 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (28972970 + index.val) (by omega)
      have evalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (28972970 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (28972970 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (28972970 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, publicNone,
        evalKNone, evalANone, selected, logicalNone, freshNone]
  | logical index =>
      have indexBound := index.isLt
      have target := logicalTarget program index
      have selected := logicalRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (29022218 + index.val)
        (by omega)
      have publicNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (29022218 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (29022218 + index.val) (by omega)
      have evalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (29022218 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_after
        (proofRange program) logicalWidth (29022218 + index.val) (by omega)
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (29022218 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, publicNone,
        evalKNone, evalANone, proofNone, selected, freshNone]
  | fresh index =>
      have indexBound := index.isLt
      norm_num [freshCount] at indexBound
      have target := freshTarget program index
      have selected := freshRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have commitNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (29022488 + index.val)
        (by omega)
      have publicNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (29022488 + index.val)
        (by omega)
      have evalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (29022488 + index.val) (by omega)
      have evalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (29022488 + index.val) (by omega)
      have proofNone := SourceRange.form?_eq_none_of_after
        (proofRange program) logicalWidth (29022488 + index.val) (by omega)
      have logicalNone := SourceRange.form?_eq_none_of_after
        (logicalRange program) logicalWidth (29022488 + index.val) (by omega)
      simp [substitution, SourceSubstitution.form?, commitNone, publicNone,
        evalKNone, evalANone, proofNone, logicalNone, selected]

/-- On every source column used by a canonical PiDEC row, the package
substitution is exactly the direct Lean source map. -/
theorem substitution_agrees_on_target
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiDECSourceSupport.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiDECDirectPlan.sourceMap geometry).form column) := by
  rcases PiDECDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, mapped⟩
  change (substitution program).form? logicalWidth column.val =
    some (match PiDECDirectPlan.classifyTarget column.val with
      | none => .empty
      | some value => value.location.form geometry)
  rw [found]
  have target :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

end NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
