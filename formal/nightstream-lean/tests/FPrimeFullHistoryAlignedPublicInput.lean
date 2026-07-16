import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedPublicInput

/-! Kernel theorem and dependency checks for the ring-aligned F' public-boundary adapter. -/

namespace tests.FPrimeFullHistoryAlignedPublicInput

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput

#check aligned_dimensions
#check dimensions_matchProduction
#check logicalProjection_preserved
#check alignedProjection_exact
#check privateSuffix_preserved
#check erase_insertPublicPadding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput.aligned_dimensions' does not depend on any axioms -/
#guard_msgs in
#print axioms aligned_dimensions

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput.erase_insertPublicPadding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms erase_insertPublicPadding

end tests.FPrimeFullHistoryAlignedPublicInput
