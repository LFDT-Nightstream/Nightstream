import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270

/-! Focused interface regression for fixed-point final ring padding. -/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270RingPadding

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

#check RingPadding.generated_rows_exact
#check RingPadding.generated_row_has_unique_offset
#check RingPaddingRefinement.expectedRow_decode_exact
#check RingPaddingRefinement.expectedDecodedRow_shape
#check RingPaddingRefinement.expectedRow_residual_eq
#check RingPaddingRefinement.generatedRowsSatisfied_iff_padding_zero
#check RingPaddingRefinement.withRingPaddingZero_satisfies
#check RingPaddingRefinement.generated_raw_row_refines

example : RingPadding.unpaddedColumns = 11437010 ∧
    RingPadding.paddingWidth = 28 ∧
    RingPadding.relationColumns = 11437038 ∧
    RingPadding.firstPaddingColumn + RingPadding.paddingWidth =
      RingPadding.relationColumns := by
  decide

example : RingPadding.rawRows.length = 28 := by
  exact RingPadding.generated_row_count

end Nightstream.Tests.FPrimeSelectiveFixedPointCarrier270RingPadding
