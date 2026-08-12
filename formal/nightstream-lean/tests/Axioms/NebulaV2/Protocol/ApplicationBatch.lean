import Nightstream.Protocol.NebulaV2.ApplicationBatch

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ApplicationBatch

open Nightstream.Protocol.NebulaV2.ApplicationBatch

#print axioms normalizedRows_flatMap_accesses
#print axioms Transition.from_terminal_is_padding
#print axioms Transition.after_valid
#print axioms Runs.count_eq_realRowCount
#print axioms Runs.splitAt
#print axioms Runs.ofCompletedExecution
#print axioms Runs.after_valid
#print axioms Batch.realRowCount_le_rowsPerFreshClaim
#print axioms Batch.after_valid
#print axioms Chain.ofRuns
#print axioms Chain.after_valid
#print axioms Chain.ofCompletedExecution

end tests.Axioms.NebulaV2ApplicationBatch
