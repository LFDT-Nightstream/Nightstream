import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchHashFrameRows

/-! Regression surface for the exact production memory-batch hash frame. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryBatchHashFrameRows

open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchHashFrameRows

#check prefix_column_values
#check claim_column_values_at
#check batch_column_values
#check input_column_values
#check inputColumns_length

end tests.NebulaV2ProductionMemoryBatchHashFrameRows
