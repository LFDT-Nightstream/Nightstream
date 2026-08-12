import Nightstream.Protocol.NebulaV2.ApplicationBatch

/-! Regression surface for exact candidate-specific application batches. -/

set_option autoImplicit false

namespace tests.NebulaV2ApplicationBatch

open Nightstream.Protocol.NebulaV2.ApplicationBatch

#check normalizedRows_flatMap_accesses
#check Transition.count_eq_rowCount
#check Transition.after_valid
#check Transition.from_terminal_is_padding
#check Runs.append
#check Runs.count_eq_realRowCount
#check Runs.count_le_length
#check Runs.after_valid
#check Runs.splitAt
#check Runs.ofActivePrefix
#check Runs.ofTerminal
#check Runs.padding
#check Runs.ofCompletedExecution
#check Runs.completed_rows_length
#check rowsPerFreshClaim_table
#check claims_rows_partition_segment
#check Batch.realRowCount_le_rowsPerFreshClaim
#check Batch.after_valid
#check Chain.rows_length
#check Chain.toRuns
#check Chain.after_valid
#check Chain.ofRuns
#check Chain.ofCompletedExecution

end tests.NebulaV2ApplicationBatch
