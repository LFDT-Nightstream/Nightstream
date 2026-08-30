import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

/-!
Proves exact source custody for the compact per-application matrix program.
Each application source resolves through one Lean-authored retained range.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open ApplicationRetainedBlocks
open ApplicationRetainedGeometry

private theorem outputEnd_le_witnessStart :
    45976 ≤ ApplicationInputs.witnessStart := by
  norm_num [ApplicationInputs.witnessStart, Spartan.privateColumnCount]

private theorem witnessStart_le_localStart (application : ApplicationProgram) :
    ApplicationInputs.witnessStart ≤ ApplicationInputs.localStart application := by
  unfold ApplicationInputs.localStart
  omega

theorem inputRange_form?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    (inputRange application).form? logicalWidth
        (ApplicationInputs.inputColumn index) =
      some ((ApplicationDirectPlan.Location.input index).form geometry) := by
  rw [ApplicationInputs.inputColumn_value]
  simpa [inputRange, ApplicationDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (inputBlock application)
      (inputStart application) ApplicationInputs.currentWordStart
      Lifecycle.Stage1.Application.stateWordCount 0 (inputFits geometry)
      (by rfl) index)

theorem witnessRange_form?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (index : Fin application.witnessWordCount) :
    (witnessRange application).form? logicalWidth
        (ApplicationInputs.witnessColumn index) =
      some ((ApplicationDirectPlan.Location.witness index).form geometry) := by
  simpa [witnessRange, ApplicationInputs.witnessColumn,
    ApplicationDirectPlan.Location.form] using
      (SourceRange.form?_ofSemantic (witnessBlock application)
        (witnessStart application) ApplicationInputs.witnessStart
        application.witnessWordCount 0 (witnessFits geometry) (by simp) index)

theorem outputRange_form?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    (outputRange application).form? logicalWidth
        (ApplicationInputs.outputColumn index) =
      some ((ApplicationDirectPlan.Location.output index).form geometry) := by
  rw [ApplicationInputs.outputColumn_value]
  simpa [outputRange, ApplicationDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (outputBlock application)
      (outputStart application) 45972
      Lifecycle.Stage1.Application.stateWordCount 0 (outputFits geometry)
      (by rfl) index)

theorem localRange_form?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (index : Fin (localCount application)) :
    (localRange application).form? logicalWidth
        (ApplicationInputs.localStart application + index.val) =
      some ((ApplicationDirectPlan.Location.localValues index).form
        geometry) := by
  simpa [localRange, ApplicationDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (localBlock application)
      (localStart application) (ApplicationInputs.localStart application)
      (localCount application) 0 (localFits geometry) (by simp) index)

/-- The compact substitution reconstructs every direct application location
and rejects all overlapping interpretations. -/
theorem substitution_location_form?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (location : ApplicationDirectPlan.Location application) :
    (substitution application).form? logicalWidth location.sourceColumn =
      some (location.form geometry) := by
  have witnessAfterOutput := outputEnd_le_witnessStart
  cases location with
  | input index =>
      have indexBound := index.isLt
      change index.val < 4 at indexBound
      have selected := inputRange_form? geometry index
      simp only [ApplicationDirectPlan.Location.sourceColumn]
      rw [ApplicationInputs.inputColumn_value]
      change (substitution application).form? logicalWidth (35 + index.val) =
        some ((ApplicationDirectPlan.Location.input index).form geometry)
      rw [ApplicationInputs.inputColumn_value] at selected
      change (inputRange application).form? logicalWidth (35 + index.val) =
        some ((ApplicationDirectPlan.Location.input index).form geometry) at selected
      have outputNone := SourceRange.form?_eq_none_of_before
        (outputRange application) logicalWidth (35 + index.val) (by
          change 35 + index.val < 45972
          omega)
      have witnessNone := SourceRange.form?_eq_none_of_before
        (witnessRange application) logicalWidth (35 + index.val) (by
          change 35 + index.val < ApplicationInputs.witnessStart
          omega)
      have localNone := SourceRange.form?_eq_none_of_before
        (localRange application) logicalWidth (35 + index.val) (by
          change 35 + index.val < ApplicationInputs.localStart application
          have localAfter := witnessStart_le_localStart application
          omega)
      simp [substitution, SourceSubstitution.form?, selected, outputNone,
        witnessNone, localNone]
  | witness index =>
      have indexBound := index.isLt
      have selected := witnessRange_form? geometry index
      simp only [ApplicationDirectPlan.Location.sourceColumn,
        ApplicationInputs.witnessColumn]
      change (witnessRange application).form? logicalWidth
        (ApplicationInputs.witnessStart + index.val) =
          some ((ApplicationDirectPlan.Location.witness index).form geometry)
        at selected
      have inputNone := SourceRange.form?_eq_none_of_after
        (inputRange application) logicalWidth
          (ApplicationInputs.witnessStart + index.val) (by
            change 35 + 4 ≤ ApplicationInputs.witnessStart + index.val
            omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange application) logicalWidth
          (ApplicationInputs.witnessStart + index.val) (by
            change 45972 + 4 ≤
              ApplicationInputs.witnessStart + index.val
            omega)
      have localNone := SourceRange.form?_eq_none_of_before
        (localRange application) logicalWidth
          (ApplicationInputs.witnessStart + index.val) (by
            change ApplicationInputs.witnessStart + index.val <
              ApplicationInputs.localStart application
            unfold ApplicationInputs.localStart
            omega)
      simp [substitution, SourceSubstitution.form?, inputNone, selected,
        outputNone, localNone]
  | output index =>
      have indexBound := index.isLt
      change index.val < 4 at indexBound
      have selected := outputRange_form? geometry index
      simp only [ApplicationDirectPlan.Location.sourceColumn]
      rw [ApplicationInputs.outputColumn_value]
      rw [ApplicationInputs.outputColumn_value] at selected
      have inputNone := SourceRange.form?_eq_none_of_after
        (inputRange application) logicalWidth (45972 + index.val) (by
          change 35 + 4 ≤ 45972 + index.val
          omega)
      have witnessNone := SourceRange.form?_eq_none_of_before
        (witnessRange application) logicalWidth (45972 + index.val) (by
          change 45972 + index.val < ApplicationInputs.witnessStart
          omega)
      have localNone := SourceRange.form?_eq_none_of_before
        (localRange application) logicalWidth (45972 + index.val) (by
          change 45972 + index.val < ApplicationInputs.localStart application
          have localAfter := witnessStart_le_localStart application
          omega)
      simp [substitution, SourceSubstitution.form?, inputNone, selected,
        witnessNone, localNone]
  | localValues index =>
      have indexBound := index.isLt
      have selected := localRange_form? geometry index
      simp only [ApplicationDirectPlan.Location.sourceColumn]
      have inputNone := SourceRange.form?_eq_none_of_after
        (inputRange application) logicalWidth
          (ApplicationInputs.localStart application + index.val) (by
            change 35 + 4 ≤
              ApplicationInputs.localStart application + index.val
            have localAfter := witnessStart_le_localStart application
            omega)
      have outputNone := SourceRange.form?_eq_none_of_after
        (outputRange application) logicalWidth
          (ApplicationInputs.localStart application + index.val) (by
            change 45972 + 4 ≤
              ApplicationInputs.localStart application + index.val
            have localAfter := witnessStart_le_localStart application
            omega)
      have witnessNone := SourceRange.form?_eq_none_of_after
        (witnessRange application) logicalWidth
          (ApplicationInputs.localStart application + index.val) (by
            change ApplicationInputs.witnessStart +
                application.witnessWordCount ≤
              ApplicationInputs.localStart application + index.val
            unfold ApplicationInputs.localStart
            omega)
      simp [substitution, SourceSubstitution.form?, inputNone, witnessNone,
        outputNone, selected]

/-- On every source column used by a canonical application row, the package
substitution is exactly the direct Lean source map. -/
theorem substitution_agrees_on_target
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (column : Fin (ApplicationRetainedBlocks.sourceWidth application))
    (support : ApplicationDirectSource.SourceAllowed application column.val) :
    (substitution application).form? logicalWidth column.val =
      some ((ApplicationDirectPlan.sourceMap geometry).form column) := by
  have complete := ApplicationDirectPlan.classifySource_complete application
    support
  cases found : ApplicationDirectPlan.classifySource application column.val with
  | none => simp [found] at complete
  | some located =>
      change (substitution application).form? logicalWidth column.val =
        some (match ApplicationDirectPlan.classifySource application column.val with
          | none => .empty
          | some value => value.location.form geometry)
      rw [found]
      simpa only [located.owns] using
        (substitution_location_form? geometry located.location)

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
