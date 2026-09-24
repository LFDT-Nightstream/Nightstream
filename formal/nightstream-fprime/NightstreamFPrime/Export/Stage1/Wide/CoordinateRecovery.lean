import NightstreamFPrime.Export.Stage1.Wide.RetainedLayout

/-! Executable inverse on retained coordinates. The new sampler interval has
no old source; no removed old coordinate is read by the projection. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CoordinateRecovery

open NightstreamFPrime.Layout
open RetainedLayout

/-- Source lookup reverses the retained interval shifts. -/
def source? (program : Program) (target : Nat) : Option Nat :=
  if target < hashEnd program then some target
  else if target < hashEnd program + (sharedEnd program - sharedStart program) then
    some (sharedStart program + (target - hashEnd program))
  else if target < commonCount program then
    some (applicationStart program +
      (target - (hashEnd program + (sharedEnd program - sharedStart program))))
  else if outputStart program ≤ target ∧ target < quotientStart program then
    some (119147994 + (target - outputStart program))
  else if quotientStart program ≤ target ∧ target < logicalWidth program then
    some (114443652 + (target - quotientStart program))
  else none

theorem source?_lt (program : Program) (target source : Nat)
    (found : source? program target = some source) : source < PerApplicationFixedPoint.logicalWidth program := by
  obtain ⟨hash, sharedStart, sharedEnd, application⟩ := boundaries program
  unfold source? at found
  rw [hash, sharedStart, sharedEnd, application, commonCount_eq] at found
  simp only [quotientStart, outputStart, commonCount_eq, logicalWidth_eq] at found
  rw [referenceWidth_eq]
  split_ifs at found <;> simp only [Option.some.injEq] at found <;> omega

theorem source?_live (program : Program) (target source : Nat)
    (found : source? program target = some source) : Live program source := by
  obtain ⟨hash, sharedStart, sharedEnd, application⟩ := boundaries program
  unfold source? at found
  rw [hash, sharedStart, sharedEnd, application, commonCount_eq] at found
  simp only [quotientStart, outputStart, commonCount_eq, logicalWidth_eq] at found
  unfold Live
  rw [hash, sharedStart, sharedEnd, application]
  split_ifs at found <;> simp only [Option.some.injEq] at found <;> omega

/-- Old coordinates copied before the direct PiRLC allocation. -/
def CommonSource (program : Program) (source : Nat) : Prop :=
  source < hashEnd program ∨
  (sharedStart program ≤ source ∧ source < sharedEnd program) ∨
  (applicationStart program ≤ source ∧ source < applicationStart program + applicationCount program)

theorem source?_common (program : Program) (target source : Nat)
    (before : target < commonCount program) (found : source? program target = some source) :
    CommonSource program source := by
  obtain ⟨hash, sharedStart, sharedEnd, application⟩ := boundaries program
  unfold source? at found
  rw [hash, sharedStart, sharedEnd, application, commonCount_eq] at found
  simp only [quotientStart, outputStart, commonCount_eq, logicalWidth_eq] at found
  rw [commonCount_eq] at before
  unfold CommonSource
  rw [hash, sharedStart, sharedEnd, application]
  split_ifs at found <;> simp only [Option.some.injEq] at found <;> omega

theorem commonSource_live (program : Program) (source : Nat)
    (common : CommonSource program source) : Live program source := by
  rcases common with hash | shared | application
  · exact Or.inl hash
  · exact Or.inr (Or.inl shared)
  · exact Or.inr (Or.inr (Or.inl application))

/-- Every copied source is placed before the direct PiRLC allocation. -/
theorem commonSource_before (program : Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program))
    (common : CommonSource program source.val) :
    (column program source (commonSource_live program source.val common)).val < commonCount program := by
  have mapped := column_mapped program source (commonSource_live program source.val common)
  obtain ⟨hash, start, stop, app⟩ := boundaries program
  unfold column? at mapped
  rw [hash, start, stop, app] at mapped
  unfold CommonSource at common
  rw [hash, start, stop, app] at common
  rw [commonCount_eq]
  split_ifs at mapped <;> simp only [Option.some.injEq] at mapped <;> omega

/-- Every mapped source recovers its own old coordinate, including both
product regions and the extension-field cell order. -/
theorem source?_column? (program : Program) (source target : Nat)
    (mapped : column? program source = some target) : source? program target = some source := by
  obtain ⟨hash, sharedStart, sharedEnd, application⟩ := boundaries program
  unfold column? at mapped
  rw [hash, sharedStart, sharedEnd, application] at mapped
  simp only [quotientStart, outputStart, commonCount_eq] at mapped
  unfold source?
  rw [hash, sharedStart, sharedEnd, application, commonCount_eq]
  simp only [quotientStart, outputStart, commonCount_eq, logicalWidth_eq]
  split_ifs at mapped <;> simp only [Option.some.injEq] at mapped
  · subst target
    rw [if_pos (by omega)]
  · subst target
    rw [if_neg (by omega), if_pos (by omega)]
    exact congrArg some (by omega)
  · subst target
    rw [if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    exact congrArg some (by omega)
  · subst target
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    exact congrArg some (by omega)
  · subst target
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    exact congrArg some (by omega)

end NightstreamFPrime.Export.Stage1.Wide.CoordinateRecovery
