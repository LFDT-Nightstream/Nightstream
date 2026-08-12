import Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ApplicationBatchCompletion

open Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion

#print axioms realRowCount_append
#print axioms realRowCount_active_rows
#print axioms realRowCount_terminal
#print axioms realRowCount_padding
#print axioms Runs.padding_inverse
#print axioms Runs.exact_shape_inverse
#print axioms completedExecution_of_exact_rows
#print axioms exact_rows_iff_completedExecution

end tests.Axioms.NebulaV2ApplicationBatchCompletion
