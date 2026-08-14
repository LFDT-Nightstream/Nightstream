import Nightstream.Protocol.Nebula.ProductionBatchedCompletion

/-! Regression surface for completed execution from the delayed lifetime. -/

set_option autoImplicit false

namespace tests.NebulaProductionBatchedCompletion

open Nightstream.Protocol.Nebula.ProductionBatchedCompletion

#check ExactCompletedRows.completedExecution
#check ExactCompletedRows.completedExecution_and_delayedSchedule

end tests.NebulaProductionBatchedCompletion
