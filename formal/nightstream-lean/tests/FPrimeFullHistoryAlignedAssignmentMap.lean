import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.AssignmentMap

/-! Compile and dependency checks for aligned assignment lowering. -/

namespace tests.FPrimeFullHistoryAlignedAssignmentMap

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap

#check getElem?_alignedIndex
#check getD_alignedIndex
#check getD_padding_zero
#check packedCoeff_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getElem?_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms getElem?_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms getD_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_padding_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms getD_padding_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.packedCoeff_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms packedCoeff_alignedIndex

end tests.FPrimeFullHistoryAlignedAssignmentMap
