import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCcsRelation

/-! Kernel theorem and dependency checks for the ring-aligned CCS relation transport. -/

namespace tests.FPrimeFullHistoryAlignedCcsRelation

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation

#check dotF_insertPublicPadding
#check matrixVector_align
#check alignStructure_wellFormed
#check rowPoint_align
#check ccsSatisfied_align_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation.ccsSatisfied_align_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms ccsSatisfied_align_iff

end tests.FPrimeFullHistoryAlignedCcsRelation
