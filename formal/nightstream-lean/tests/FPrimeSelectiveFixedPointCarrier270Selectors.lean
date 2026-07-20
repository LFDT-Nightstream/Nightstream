import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270

/-! Focused interface regression for fixed-point selector rows. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270Selectors

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check Selectors.generated_rows_exact
#check Selectors.generated_row_has_unique_owner
#check SelectorRefinement.expectedSelectorRow_decode_exact
#check SelectorRefinement.expectedTotalRow_decode_exact
#check SelectorRefinement.generated_raw_row_decodes
#check SelectorRefinement.expectedSelectorResidual_eq
#check SelectorRefinement.expectedTotalResidual_eq
#check SelectorRefinement.expectedSelectorRow_satisfied_iff_boolean
#check SelectorRefinement.expectedTotalRow_satisfied_iff_total
#check SelectorRefinement.generatedRowsSatisfied_iff
#check SelectorRefinement.withUnitSelectors_satisfies

example : Selectors.selectorCount = 3 ∧ Selectors.selectorStart = 270 := by
  decide

example : Selectors.rawRows.length = 4 := by
  exact Selectors.generated_row_count

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270Selectors
