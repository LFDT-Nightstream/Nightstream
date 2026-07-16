import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ColumnMap

/-! Compile and dependency checks for the aligned compiler column map. -/

namespace tests.FPrimeFullHistoryAlignedColumnMap

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap

#check alignedIndex_injective
#check alignColumn_injective
#check unalignIndex?_alignedIndex
#check unalignIndex?_eq_none_iff
#check alignedIndex_of_unalignIndex?_eq_some
#check unalignedIndex_lt
#check packedFlatIndex
#check boundary_coordinates
#check fixedCarrier_blockCounts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedIndex_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignColumn_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms alignColumn_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms unalignIndex?_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_eq_none_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms unalignIndex?_eq_none_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_of_unalignIndex?_eq_some' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms alignedIndex_of_unalignIndex?_eq_some

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignedIndex_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms unalignedIndex_lt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.packedFlatIndex' depends on axioms: [propext] -/
#guard_msgs in
#print axioms packedFlatIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.boundary_coordinates' does not depend on any axioms -/
#guard_msgs in
#print axioms boundary_coordinates

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.fixedCarrier_blockCounts' does not depend on any axioms -/
#guard_msgs in
#print axioms fixedCarrier_blockCounts

end tests.FPrimeFullHistoryAlignedColumnMap
