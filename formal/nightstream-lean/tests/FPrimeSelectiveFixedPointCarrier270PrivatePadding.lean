import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270

/-! Focused interface regression for fixed-point private-alignment padding. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PrivatePadding

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check PrivatePadding.generated_rows_exact
#check PrivatePadding.generated_row_has_unique_offset
#check PrivatePaddingRefinement.expectedRow_decode_exact
#check PrivatePaddingRefinement.expectedDecodedRow_shape
#check PrivatePaddingRefinement.expectedRow_residual_eq
#check PrivatePaddingRefinement.generatedRowsSatisfied_iff_padding_zero
#check PrivatePaddingRefinement.withPrivatePaddingZero_satisfies
#check PrivatePaddingRefinement.generated_raw_row_refines

example : PrivatePadding.paddingWidth = 38 ∧
    PrivatePadding.firstPaddingColumn = 273 ∧
    PrivatePadding.firstPaddingColumn + PrivatePadding.paddingWidth = 311 := by
  decide

example : PrivatePadding.rawRows.length = 38 := by
  exact PrivatePadding.generated_row_count

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270PrivatePadding
