import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryXOutSpongeReceipts

/-!
Focused elaboration boundary for the source-computed and artifact-checked
plain-state XOut Poseidon2 sponge receipts.
-/

namespace NightstreamTests.FPrimeFullHistoryXOutSpongeReceipts

open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts

#check EmissionReceipt.traceRows_length
#check EmissionReceipt.rowIndices_nodup
#check EmissionReceipt.allocatedColumns_nodup
#check EmissionReceipt.row_column_conservation
#check sourceProgram_eq_generated
#check inputFields_eq
#check physicalCost_eq
#check baseSchedule_exact
#check priorSchedule_exact
#check recursiveOutputSchedule_exact
#check baseReceipt
#check priorReceipt
#check recursiveOutputReceipt
#check baseRows_exact_cost
#check priorRows_exact_cost
#check recursiveOutputRows_exact_cost
#check base_conservation
#check prior_conservation
#check recursiveOutput_conservation
#check pureExecutions_equal

end NightstreamTests.FPrimeFullHistoryXOutSpongeReceipts
