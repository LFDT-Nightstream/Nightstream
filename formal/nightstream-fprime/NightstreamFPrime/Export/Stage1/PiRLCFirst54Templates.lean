import NightstreamFPrime.Export.Stage1.CompactRows
import NightstreamFPrime.Gadgets.Sampling.First54

/-!
Owns the four normalized compact-template families for the PiRLC `First54`
selector.

Round zero has verifier-fixed prior position and output values. Later rounds
read the preceding selector columns. The templates keep these cases separate
and reuse each indexed slot across all later rounds and all 17 samplers.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54Templates

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout

def firstPositionInputCount : Nat := 2
def firstPositionOutputInput : Nat := 1

def laterPositionInputCount : Nat := 57
def laterPositionPriorStart : Nat := 1
def laterPositionOutputInput : Nat := 56

def firstValueInputCount : Nat := 3
def firstValueOutputInput : Nat := 2

def laterValueInputCount : Nat := 112
def laterValuePriorPositionStart : Nat := 2
def laterValuePriorOutputStart : Nat := 57
def laterValueOutputInput : Nat := 111

def acceptedFromReject (rejectInput : Nat) : Expr :=
  1 - Expr.var rejectInput

def firstPositionInterface : First54Step.Interface where
  accepted := fun _ => acceptedFromReject 0
  prior := fun _ slot => First54.initialPosition slot

def laterPositionInterface : First54Step.Interface where
  accepted := fun _ => acceptedFromReject 0
  prior := fun _ slot => Expr.var (laterPositionPriorStart + slot.val)

def firstValueInterface : First54ValueStep.Interface where
  accepted := fun _ => acceptedFromReject 0
  symbol := fun _ => Expr.var 1
  priorPosition := fun _ slot => First54.initialPosition slot
  priorOutput := fun _ _ => 0

def laterValueInterface : First54ValueStep.Interface where
  accepted := fun _ => acceptedFromReject 0
  symbol := fun _ => Expr.var 1
  priorPosition := fun _ slot =>
    Expr.var (laterValuePriorPositionStart + slot.val)
  priorOutput := fun _ slot =>
    Expr.var (laterValuePriorOutputStart + slot.val)

def firstPositionRecipe (slot : Fin First54Step.slotCount) : Expr :=
  First54Step.recipe firstPositionInterface 0 slot

def laterPositionRecipe (slot : Fin First54Step.slotCount) : Expr :=
  First54Step.recipe laterPositionInterface 0 slot

def firstValueRecipe (slot : Fin First54ValueStep.outputCount) : Expr :=
  First54ValueStep.recipe firstValueInterface 0 slot

def laterValueRecipe (slot : Fin First54ValueStep.outputCount) : Expr :=
  First54ValueStep.recipe laterValueInterface 0 slot

def firstPositionTemplate (slot : Fin First54Step.slotCount) :
    CompactRowTemplate :=
  CompactRows.compactConstraintTemplate firstPositionInputCount
    firstPositionOutputInput (firstPositionRecipe slot)

def laterPositionTemplate (slot : Fin First54Step.slotCount) :
    CompactRowTemplate :=
  CompactRows.compactConstraintTemplate laterPositionInputCount
    laterPositionOutputInput (laterPositionRecipe slot)

def firstValueTemplate (slot : Fin First54ValueStep.outputCount) :
    CompactRowTemplate :=
  CompactRows.compactConstraintTemplate firstValueInputCount
    firstValueOutputInput (firstValueRecipe slot)

def laterValueTemplate (slot : Fin First54ValueStep.outputCount) :
    CompactRowTemplate :=
  CompactRows.compactConstraintTemplate laterValueInputCount
    laterValueOutputInput (laterValueRecipe slot)

set_option maxRecDepth 100000 in -- fixed-size: 55 position templates
@[simp] theorem firstPosition_constraintFreshCount
    (slot : Fin First54Step.slotCount) :
    R1CS.constraintFreshCount
        (Expr.var firstPositionOutputInput - firstPositionRecipe slot) = 0 := by
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 55 position templates
@[simp] theorem laterPosition_constraintFreshCount
    (slot : Fin First54Step.slotCount) :
    R1CS.constraintFreshCount
        (Expr.var laterPositionOutputInput - laterPositionRecipe slot) =
      (if slot.val = 0 then 0 else if slot.val = 54 then 3 else 6) := by
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 54 value templates
@[simp] theorem firstValue_constraintFreshCount
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (Expr.var firstValueOutputInput - firstValueRecipe slot) = 4 := by
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 54 value templates
@[simp] theorem laterValue_constraintFreshCount
    (slot : Fin First54ValueStep.outputCount) :
    R1CS.constraintFreshCount
        (Expr.var laterValueOutputInput - laterValueRecipe slot) = 4 := by
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 55 position templates
@[simp] theorem firstPositionTemplate_rows_length
    (slot : Fin First54Step.slotCount) :
    (firstPositionTemplate slot).rows.length = 1 := by
  simp only [firstPositionTemplate,
    CompactRows.compactConstraintTemplate_rows_length]
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 55 position templates
@[simp] theorem laterPositionTemplate_rows_length
    (slot : Fin First54Step.slotCount) :
    (laterPositionTemplate slot).rows.length =
      (if slot.val = 0 then 1 else if slot.val = 54 then 4 else 7) := by
  simp only [laterPositionTemplate,
    CompactRows.compactConstraintTemplate_rows_length]
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 54 value templates
@[simp] theorem firstValueTemplate_rows_length
    (slot : Fin First54ValueStep.outputCount) :
    (firstValueTemplate slot).rows.length = 5 := by
  simp only [firstValueTemplate,
    CompactRows.compactConstraintTemplate_rows_length]
  fin_cases slot <;> rfl

set_option maxRecDepth 100000 in -- fixed-size: 54 value templates
@[simp] theorem laterValueTemplate_rows_length
    (slot : Fin First54ValueStep.outputCount) :
    (laterValueTemplate slot).rows.length = 5 := by
  simp only [laterValueTemplate,
    CompactRows.compactConstraintTemplate_rows_length]
  fin_cases slot <;> rfl

def firstPositionTemplates : List CompactRowTemplate :=
  (List.finRange First54Step.slotCount).map firstPositionTemplate

def laterPositionTemplates : List CompactRowTemplate :=
  (List.finRange First54Step.slotCount).map laterPositionTemplate

def firstValueTemplates : List CompactRowTemplate :=
  (List.finRange First54ValueStep.outputCount).map firstValueTemplate

def laterValueTemplates : List CompactRowTemplate :=
  (List.finRange First54ValueStep.outputCount).map laterValueTemplate

def templates : List CompactRowTemplate :=
  firstPositionTemplates ++
    (laterPositionTemplates ++
      (firstValueTemplates ++ laterValueTemplates))

def firstPositionTemplateIndex (slot : Nat) : Nat := slot
def laterPositionTemplateIndex (slot : Nat) : Nat :=
  First54Step.slotCount + slot
def firstValueTemplateIndex (slot : Nat) : Nat :=
  2 * First54Step.slotCount + slot
def laterValueTemplateIndex (slot : Nat) : Nat :=
  2 * First54Step.slotCount + First54ValueStep.outputCount + slot

private theorem map_finRange_getElem? {count : Nat} {Alpha : Type}
    (entry : Fin count → Alpha) (slot : Nat) (bound : slot < count) :
    ((List.finRange count).map entry)[slot]? =
      some (entry ⟨slot, bound⟩) := by
  rw [List.getElem?_eq_getElem (by simpa using bound)]
  simp

theorem firstPositionTemplate_getElem? (slot : Nat)
    (bound : slot < First54Step.slotCount) :
    templates[firstPositionTemplateIndex slot]? =
      some (firstPositionTemplate ⟨slot, bound⟩) := by
  unfold templates firstPositionTemplateIndex
  rw [List.getElem?_append_left (by
    simpa [firstPositionTemplates] using bound)]
  exact map_finRange_getElem? firstPositionTemplate slot bound

theorem laterPositionTemplate_getElem? (slot : Nat)
    (bound : slot < First54Step.slotCount) :
    templates[laterPositionTemplateIndex slot]? =
      some (laterPositionTemplate ⟨slot, bound⟩) := by
  have firstLen : firstPositionTemplates.length = First54Step.slotCount := by
    simp [firstPositionTemplates]
  unfold templates laterPositionTemplateIndex
  rw [List.getElem?_append_right (by
    rw [firstLen]
    omega)]
  rw [firstLen, show First54Step.slotCount + slot -
      First54Step.slotCount = slot by omega]
  rw [List.getElem?_append_left (by
    simpa [laterPositionTemplates] using bound)]
  exact map_finRange_getElem? laterPositionTemplate slot bound

theorem firstValueTemplate_getElem? (slot : Nat)
    (bound : slot < First54ValueStep.outputCount) :
    templates[firstValueTemplateIndex slot]? =
      some (firstValueTemplate ⟨slot, bound⟩) := by
  have firstLen : firstPositionTemplates.length = First54Step.slotCount := by
    simp [firstPositionTemplates]
  have laterLen : laterPositionTemplates.length = First54Step.slotCount := by
    simp [laterPositionTemplates]
  unfold templates firstValueTemplateIndex
  rw [List.getElem?_append_right (by
    rw [firstLen]
    omega)]
  rw [firstLen, show 2 * First54Step.slotCount + slot -
      First54Step.slotCount =
      First54Step.slotCount + slot by
    omega]
  rw [List.getElem?_append_right (by
    rw [laterLen]
    omega)]
  rw [laterLen, show First54Step.slotCount + slot -
      First54Step.slotCount = slot by omega]
  rw [List.getElem?_append_left (by
    simpa [firstValueTemplates] using bound)]
  exact map_finRange_getElem? firstValueTemplate slot bound

theorem laterValueTemplate_getElem? (slot : Nat)
    (bound : slot < First54ValueStep.outputCount) :
    templates[laterValueTemplateIndex slot]? =
      some (laterValueTemplate ⟨slot, bound⟩) := by
  have firstLen : firstPositionTemplates.length = First54Step.slotCount := by
    simp [firstPositionTemplates]
  have laterLen : laterPositionTemplates.length = First54Step.slotCount := by
    simp [laterPositionTemplates]
  have valueLen : firstValueTemplates.length = First54ValueStep.outputCount := by
    simp [firstValueTemplates]
  unfold templates laterValueTemplateIndex
  rw [List.getElem?_append_right (by
    rw [firstLen]
    omega)]
  rw [firstLen]
  rw [show 2 * First54Step.slotCount + First54ValueStep.outputCount + slot -
      First54Step.slotCount =
        First54Step.slotCount + First54ValueStep.outputCount + slot by omega]
  rw [List.getElem?_append_right (by
    rw [laterLen]
    omega)]
  rw [laterLen, show First54Step.slotCount +
      First54ValueStep.outputCount + slot - First54Step.slotCount =
        First54ValueStep.outputCount + slot by omega]
  rw [List.getElem?_append_right (by
    rw [valueLen]
    omega)]
  rw [valueLen, show First54ValueStep.outputCount + slot -
      First54ValueStep.outputCount = slot by omega]
  exact map_finRange_getElem? laterValueTemplate slot bound

private theorem positionRecipe_varsBelow (interface : First54Step.Interface)
    (offset bound : Nat) (slot : Fin First54Step.slotCount)
    (acceptedBelow : (interface.accepted offset).VarsBelow bound)
    (priorBelow : ∀ current, (interface.prior offset current).VarsBelow bound) :
    (First54Step.recipe interface offset slot).VarsBelow bound := by
  unfold First54Step.recipe
  split
  · exact Expr.VarsBelow.mul _ _ _ (priorBelow slot)
      (Expr.VarsBelow.sub _ _ _ trivial acceptedBelow)
  · split
    · exact Expr.VarsBelow.add _ _ _ (priorBelow slot)
        (Expr.VarsBelow.mul _ _ _
          (priorBelow (First54Step.previousSlot slot (by omega)))
          acceptedBelow)
    · exact Expr.VarsBelow.add _ _ _
        (Expr.VarsBelow.mul _ _ _ (priorBelow slot)
          (Expr.VarsBelow.sub _ _ _ trivial acceptedBelow))
        (Expr.VarsBelow.mul _ _ _
          (priorBelow (First54Step.previousSlot slot (by omega)))
          acceptedBelow)

private theorem valueRecipe_varsBelow
    (interface : First54ValueStep.Interface) (offset bound : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (acceptedBelow : (interface.accepted offset).VarsBelow bound)
    (symbolBelow : (interface.symbol offset).VarsBelow bound)
    (priorPositionBelow : ∀ current,
      (interface.priorPosition offset current).VarsBelow bound)
    (priorOutputBelow : ∀ current,
      (interface.priorOutput offset current).VarsBelow bound) :
    (First54ValueStep.recipe interface offset slot).VarsBelow bound := by
  unfold First54ValueStep.recipe
  exact Expr.VarsBelow.add _ _ _ (priorOutputBelow slot)
    (Expr.VarsBelow.mul _ _ _
      (Expr.VarsBelow.mul _ _ _
        (priorPositionBelow (First54ValueStep.positionSlot slot))
        acceptedBelow)
      symbolBelow)

theorem firstPosition_constraint_varsBelow
    (slot : Fin First54Step.slotCount) :
    (Expr.var firstPositionOutputInput - firstPositionRecipe slot).VarsBelow
      firstPositionInputCount := by
  apply Expr.VarsBelow.sub
  · change firstPositionOutputInput < firstPositionInputCount
    norm_num [firstPositionOutputInput, firstPositionInputCount]
  · apply positionRecipe_varsBelow firstPositionInterface 0
        firstPositionInputCount slot
    · exact Expr.VarsBelow.sub _ _ _ trivial (by
        change 0 < firstPositionInputCount
        norm_num [firstPositionInputCount])
    · intro current
      change (First54.initialPosition current).VarsBelow
        firstPositionInputCount
      unfold First54.initialPosition
      split <;> trivial

theorem laterPosition_constraint_varsBelow
    (slot : Fin First54Step.slotCount) :
    (Expr.var laterPositionOutputInput - laterPositionRecipe slot).VarsBelow
      laterPositionInputCount := by
  apply Expr.VarsBelow.sub
  · change laterPositionOutputInput < laterPositionInputCount
    norm_num [laterPositionOutputInput, laterPositionInputCount]
  · apply positionRecipe_varsBelow laterPositionInterface 0
        laterPositionInputCount slot
    · exact Expr.VarsBelow.sub _ _ _ trivial (by
        change 0 < laterPositionInputCount
        norm_num [laterPositionInputCount])
    · intro current
      change laterPositionPriorStart + current.val < laterPositionInputCount
      have bounded := current.isLt
      norm_num [laterPositionPriorStart, laterPositionInputCount,
        First54Step.slotCount] at bounded ⊢
      omega

theorem firstValue_constraint_varsBelow
    (slot : Fin First54ValueStep.outputCount) :
    (Expr.var firstValueOutputInput - firstValueRecipe slot).VarsBelow
      firstValueInputCount := by
  apply Expr.VarsBelow.sub
  · change firstValueOutputInput < firstValueInputCount
    norm_num [firstValueOutputInput, firstValueInputCount]
  · apply valueRecipe_varsBelow firstValueInterface 0 firstValueInputCount
        slot
    · exact Expr.VarsBelow.sub _ _ _ trivial (by
        change 0 < firstValueInputCount
        norm_num [firstValueInputCount])
    · change 1 < firstValueInputCount
      norm_num [firstValueInputCount]
    · intro current
      change (First54.initialPosition current).VarsBelow firstValueInputCount
      unfold First54.initialPosition
      split <;> trivial
    · intro current
      trivial

theorem laterValue_constraint_varsBelow
    (slot : Fin First54ValueStep.outputCount) :
    (Expr.var laterValueOutputInput - laterValueRecipe slot).VarsBelow
      laterValueInputCount := by
  apply Expr.VarsBelow.sub
  · change laterValueOutputInput < laterValueInputCount
    norm_num [laterValueOutputInput, laterValueInputCount]
  · apply valueRecipe_varsBelow laterValueInterface 0 laterValueInputCount
        slot
    · exact Expr.VarsBelow.sub _ _ _ trivial (by
        change 0 < laterValueInputCount
        norm_num [laterValueInputCount])
    · change 1 < laterValueInputCount
      norm_num [laterValueInputCount]
    · intro current
      change laterValuePriorPositionStart + current.val < laterValueInputCount
      have bounded := current.isLt
      norm_num [laterValuePriorPositionStart, laterValueInputCount,
        First54Step.slotCount] at bounded ⊢
      omega
    · intro current
      change laterValuePriorOutputStart + current.val < laterValueInputCount
      have bounded := current.isLt
      norm_num [laterValuePriorOutputStart, laterValueInputCount,
        First54ValueStep.outputCount] at bounded ⊢
      omega

@[simp] theorem firstPositionTemplates_length :
    firstPositionTemplates.length = 55 := by
  simp [firstPositionTemplates, First54Step.slotCount]

@[simp] theorem laterPositionTemplates_length :
    laterPositionTemplates.length = 55 := by
  simp [laterPositionTemplates, First54Step.slotCount]

@[simp] theorem firstValueTemplates_length :
    firstValueTemplates.length = 54 := by
  simp [firstValueTemplates, First54ValueStep.outputCount]

@[simp] theorem laterValueTemplates_length :
    laterValueTemplates.length = 54 := by
  simp [laterValueTemplates, First54ValueStep.outputCount]

@[simp] theorem templates_length : templates.length = 218 := by
  simp [templates]

theorem firstPositionTemplates_rowCount :
    (firstPositionTemplates.map (fun template => template.rows.length)).sum =
      55 := by
  simp [firstPositionTemplates, Function.comp_def, First54Step.slotCount]

theorem laterPositionTemplates_rowCount :
    (laterPositionTemplates.map (fun template => template.rows.length)).sum =
      376 := by
  rfl

theorem firstValueTemplates_rowCount :
    (firstValueTemplates.map (fun template => template.rows.length)).sum =
      270 := by
  simp [firstValueTemplates, Function.comp_def,
    First54ValueStep.outputCount]

theorem laterValueTemplates_rowCount :
    (laterValueTemplates.map (fun template => template.rows.length)).sum =
      270 := by
  simp [laterValueTemplates, Function.comp_def,
    First54ValueStep.outputCount]

end NightstreamFPrime.Export.Stage1.PiRLCFirst54Templates
