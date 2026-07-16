import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.MatrixMap

/-! Compile and dependency checks for aligned matrix-row lowering. -/

namespace tests.FPrimeFullHistoryAlignedMatrixMap

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap

#check compiledRowValue_old
#check compiledRowValue_padding
#check compiledRowValue_cases
#check alignRow_getD_eq_compiledRowValue

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_old' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms compiledRowValue_old

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_padding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms compiledRowValue_padding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_cases' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms compiledRowValue_cases

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.alignRow_getD_eq_compiledRowValue' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms alignRow_getD_eq_compiledRowValue

end tests.FPrimeFullHistoryAlignedMatrixMap
