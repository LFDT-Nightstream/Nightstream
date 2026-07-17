import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.SelectiveLayout

/-! Focused checks for the typed selective-compiler carrier layout. -/

namespace tests.FPrimeFullHistorySelectiveLayout

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout

#check exact_layout
#check selectorColumn_bounds
#check selector_not_in_public_carrier
#check legacy_selector_in_public_carrier
#check privateAlignment_range

end tests.FPrimeFullHistorySelectiveLayout
