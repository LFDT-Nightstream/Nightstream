import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.SuccessorStateBindingRowsFor

/-! Regression surface for the exponent-indexed F-prime successor rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionSuccessorStateBindingRowsFor

open Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBindingRowsFor

#check section_offsets_exact
#check runningNativeFields_length
#check successorFields_length
#check fieldValues_eq_successorFrame
#check successorPermutationCount_25
#check builder_entries_length
#check rows_length_exact
#check rows_imply_outputState
#check rows_imply_outputDigest_lane

example : successorPermutationCount 25 = 20878 :=
  successorPermutationCount_25

end tests.NebulaV2ProductionSuccessorStateBindingRowsFor
