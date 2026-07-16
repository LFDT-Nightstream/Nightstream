import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedNifsRepair

/-! Compile and dependency surface for the fail-closed aligned NIFS repair. -/

namespace tests.FPrimeFullHistoryAlignedNifsRepair

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair

#check fixedCarrier_paperDimensions
#check alignedFreshStatement_holds
#check alignedNifsTransition_complete
#check fixedCarrierTail_not_normBounded
#check fixedCarrierTail_not_alignedCCS
#check fixedCarrierTail_not_alignedCE

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrier_paperDimensions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms fixedCarrier_paperDimensions

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedFreshStatement_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedFreshStatement_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedNifsTransition_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedNifsTransition_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_normBounded' depends on axioms: [propext] -/
#guard_msgs in
#print axioms fixedCarrierTail_not_normBounded

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCCS' depends on axioms: [propext] -/
#guard_msgs in
#print axioms fixedCarrierTail_not_alignedCCS

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCE' depends on axioms: [propext] -/
#guard_msgs in
#print axioms fixedCarrierTail_not_alignedCE

end tests.FPrimeFullHistoryAlignedNifsRepair
