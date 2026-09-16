import NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Templates

/-!
Each canonical PiRLC compact recipe reads only normalized inputs before its
output. These structural bounds do not depend on field values or physical
column geometry.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open PiRLCCombinationTemplates
open PiRLCFirst54Templates

theorem combination_outputRecipe (firstSource : Bool) (lane : Fin ringDegree) :
    (PiRLCCombinationTemplates.outputRecipe firstSource lane).VarsBelow
      PiRLCCombinationTemplates.outputInput := by
  unfold PiRLCCombinationTemplates.outputRecipe
  apply Expr.VarsBelow.add
  · unfold PiRLCCombinationTemplates.prior
    split
    · trivial
    · change priorInput < outputInput
      norm_num [priorInput, outputInput]
  · apply CombinationStep.mulExpr_varsBelow
    · intro current
      unfold PiRLCCombinationTemplates.challenge
      apply Expr.VarsBelow.sub
      · change challengeInputStart + current.val < outputInput
        have currentBound : current.val < 54 := by
          simpa [ringDegree] using current.isLt
        norm_num [challengeInputStart, outputInput]
        omega
      · trivial
    · intro current
      unfold PiRLCCombinationTemplates.value
      change valueInputStart + current.val < outputInput
      have currentBound : current.val < 54 := by
        simpa [ringDegree] using current.isLt
      norm_num [valueInputStart, outputInput]
      omega

theorem firstPosition_recipe (slot : Fin First54Step.slotCount) :
    (firstPositionRecipe slot).VarsBelow firstPositionOutputInput := by
  apply positionRecipe_varsBelow firstPositionInterface 0 firstPositionOutputInput slot
  · exact Expr.VarsBelow.sub _ _ _ trivial (by
      change 0 < firstPositionOutputInput
      norm_num [firstPositionOutputInput])
  · intro current
    change (First54.initialPosition current).VarsBelow firstPositionOutputInput
    unfold First54.initialPosition
    split <;> trivial

theorem laterPosition_recipe (slot : Fin First54Step.slotCount) :
    (laterPositionRecipe slot).VarsBelow laterPositionOutputInput := by
  apply positionRecipe_varsBelow laterPositionInterface 0 laterPositionOutputInput slot
  · exact Expr.VarsBelow.sub _ _ _ trivial (by
      change 0 < laterPositionOutputInput
      norm_num [laterPositionOutputInput])
  · intro current
    change laterPositionPriorStart + current.val < laterPositionOutputInput
    have bounded := current.isLt
    norm_num [laterPositionPriorStart, laterPositionOutputInput,
      First54Step.slotCount] at bounded ⊢
    omega

theorem firstValue_recipe (slot : Fin First54ValueStep.outputCount) :
    (firstValueRecipe slot).VarsBelow firstValueOutputInput := by
  apply valueRecipe_varsBelow firstValueInterface 0 firstValueOutputInput slot
  · exact Expr.VarsBelow.sub _ _ _ trivial (by
      change 0 < firstValueOutputInput
      norm_num [firstValueOutputInput])
  · change 1 < firstValueOutputInput
    norm_num [firstValueOutputInput]
  · intro current
    change (First54.initialPosition current).VarsBelow firstValueOutputInput
    unfold First54.initialPosition
    split <;> trivial
  · intro current
    trivial

theorem laterValue_recipe (slot : Fin First54ValueStep.outputCount) :
    (laterValueRecipe slot).VarsBelow laterValueOutputInput := by
  apply valueRecipe_varsBelow laterValueInterface 0 laterValueOutputInput slot
  · exact Expr.VarsBelow.sub _ _ _ trivial (by
      change 0 < laterValueOutputInput
      norm_num [laterValueOutputInput])
  · change 1 < laterValueOutputInput
    norm_num [laterValueOutputInput]
  · intro current
    change laterValuePriorPositionStart + current.val < laterValueOutputInput
    have bounded := current.isLt
    norm_num [laterValuePriorPositionStart, laterValueOutputInput,
      First54Step.slotCount] at bounded ⊢
    omega
  · intro current
    change laterValuePriorOutputStart + current.val < laterValueOutputInput
    have bounded := current.isLt
    norm_num [laterValuePriorOutputStart, laterValueOutputInput,
      First54ValueStep.outputCount] at bounded ⊢
    omega

end NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope
