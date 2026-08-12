import Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion

/-! Regression surface for reverse completed-execution extraction. -/

set_option autoImplicit false

namespace tests.NebulaV2ApplicationBatchCompletion

open Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion

#check realRowCount_append
#check realRowCount_active_rows
#check realRowCount_terminal
#check realRowCount_padding
#check Runs.padding_inverse
#check Runs.after_terminal_is_padding
#check Runs.terminal_shape_inverse
#check Runs.exact_shape_inverse
#check completedExecution_of_exact_rows
#check ExactCompletedRun
#check exactCompletedRun_of_terminal_run
#check completedExecution_of_terminal_run
#check ExactCompletedRun.accessesExact
#check exact_rows_iff_completedExecution

end tests.NebulaV2ApplicationBatchCompletion
