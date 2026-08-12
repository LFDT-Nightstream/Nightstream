import Nightstream.Protocol.NebulaV2.ProductionBatchedCompletion

/-! Regression surface for completed execution from the delayed lifetime. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchedCompletion

open Nightstream.Protocol.NebulaV2.ProductionBatchedCompletion

#check ExactCompletedRows.completedExecution
#check ExactCompletedRows.completedExecution_and_delayedSchedule

end tests.NebulaV2ProductionBatchedCompletion
