import NightstreamFPrime.Export.Stage1.StoredCompactOutput
import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocationOrigin
import NightstreamFPrime.Export.Stage1.PiRLCCompactRecipeScope

/-!
The canonical PiRLC product records satisfy the structural conditions for
stored output-only execution, including the application column shift.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPiRLCCombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open PiRLCCombinationScratchGeometry (scratchStart scratchEnd)
open StoredExecutionSupport (Agree)

/-- Each shifted canonical product can omit its full R1CS scratch execution.
Success/rejection and every value outside global scratch agree, for arbitrary
input arrays of equal size that already agree outside that interval. -/
theorem shifted_agree (context : PerApplicationCachedShift.Context)
    (descriptor : PiRLCProductSchedule.Descriptor) (left right : Array F)
    (agree : Agree (fun column => column < scratchStart ∨ scratchEnd ≤ column) left right) :
    let invocation := PerApplicationCachedShift.shiftCompactRowInvocation context
      descriptor.compactInvocation
    let template := PiRLCCombinationTemplates.template
      (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane
    Option.Rel (Agree (fun column => column < scratchStart ∨ scratchEnd ≤ column))
      (StoredCompactRowExecution.execute (compactInputColumn invocation.inputRanges)
        invocation.localStart template left)
      (StoredCompactOutput.execute (compactInputColumn invocation.inputRanges)
        invocation.localStart template right) := by
  let invocation := PerApplicationCachedShift.shiftCompactRowInvocation context
    descriptor.compactInvocation
  let inputs := compactInputColumn invocation.inputRanges
  let recipe := PiRLCCombinationTemplates.outputRecipe
    (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane
  have contained := PiRLCCombinationInvocationOrigin.shifted_scratch_contained context descriptor
  change scratchStart ≤ invocation.localStart ∧ invocation.localStart +
    PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ scratchEnd at contained
  have inputScope := PiRLCCombinationInvocationOrigin.shifted_inputs_outside_scratch context descriptor
  have recipeBelow := PiRLCCompactRecipeScope.combination_outputRecipe
    (PiRLCCombinationInvocations.firstSource descriptor.source.val) descriptor.lane
  apply StoredCompactOutput.compactTemplate_agree
    (fun column => column < scratchStart ∨ scratchEnd ≤ column)
    PiRLCCombinationTemplates.inputCount PiRLCCombinationTemplates.outputInput
    invocation.localStart inputs recipe left right
  · rw [PiRLCCombinationInvocationOrigin.shifted_localStart]
    exact PiRLCCombinationScratchGeometry.localStart_ge descriptor
  · intro input bound
    have outside := inputScope input bound
    change inputs input < scratchStart ∨ scratchEnd ≤ inputs input at outside
    change inputs input < invocation.localStart ∨ invocation.localStart +
      PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ inputs input
    rcases outside with earlier | later
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  · decide
  · exact recipeBelow
  · exact PiRLCCombinationInvocationOrigin.shifted_output_distinct context descriptor
  · apply Expr.VarsSatisfy.mono recipe
      ((Expr.varsSatisfy_lt_iff_varsBelow recipe
        PiRLCCombinationTemplates.outputInput).mpr recipeBelow)
    intro input bound
    exact inputScope input (by
      change input < 109 at bound
      change input < 110
      omega)
  · intro column outside
    change column < invocation.localStart ∨ invocation.localStart +
      PiRLCCombinationScratchGeometry.scratchCount descriptor ≤ column
    rcases outside with earlier | later
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  · exact agree

end NightstreamFPrime.Export.Stage1.StoredPiRLCCombination
