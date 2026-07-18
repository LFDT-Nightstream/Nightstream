import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding

/-! Focused checks for typed selective-CCS public-padding semantics. -/

namespace tests.FPrimeFullHistorySelectiveCcsPadding

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Necessity

#check constantColumn_ne_padding
#check paddingCarrierColumn_injective
#check zeroPinHolds_iff
#check allZeroPinsHold_iff_fixedPublicPadding
#check canonicalAssignment_complete
#check oneAtPadding_others_hold
#check oneAtPadding_violates_fixedPublicPadding
#check eachRawPaddingCheck_necessary

end tests.FPrimeFullHistorySelectiveCcsPadding
