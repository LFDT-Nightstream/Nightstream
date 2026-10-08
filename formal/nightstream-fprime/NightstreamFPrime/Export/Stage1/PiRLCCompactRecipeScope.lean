import NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates

/-!
Each canonical PiRLC compact recipe reads only normalized inputs before its
output. These structural bounds do not depend on field values or physical
column geometry.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open PiRLCCombinationTemplates

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
          simp [ringDegree]
        norm_num [challengeInputStart, outputInput]
        omega
      · trivial
    · intro current
      unfold PiRLCCombinationTemplates.value
      change valueInputStart + current.val < outputInput
      have currentBound : current.val < 54 := by
        simp [ringDegree]
      norm_num [valueInputStart, outputInput]
      omega

end NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope
