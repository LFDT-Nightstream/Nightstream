import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270

/-! Focused interface regression for fixed-point public padding. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PublicPadding

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check PublicPadding.generated_rows_exact
#check PublicPadding.generated_row_has_unique_offset
#check PublicPaddingRefinement.expectedRow_decode_exact
#check PublicPaddingRefinement.expectedDecodedRow_shape
#check PublicPaddingRefinement.expectedRow_residual_eq
#check PublicPaddingRefinement.generatedRowsSatisfied_iff_padding_zero
#check PublicPaddingRefinement.withPublicPaddingZero_satisfies
#check PublicPaddingRefinement.generatedRowsSatisfied_iff_fixedPublicPadding
#check PublicPaddingRefinement.generatedRowsSatisfied_of_typedAssignment
#check PublicPaddingRefinement.generated_raw_row_refines

example : PublicPadding.paddingWidth = 13 ∧
    PublicPadding.firstPaddingColumn = 257 ∧
    PublicPadding.firstPaddingColumn + PublicPadding.paddingWidth = 270 := by
  decide

example : PublicPadding.rawRows.length = 13 := by
  exact PublicPadding.generated_row_count

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PublicPadding
