import Nightstream.Implementation.Nebula.Production.Memory.BatchHashFrameRows

/-! Regression surface for the exact production memory-batch hash frame. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryBatchHashFrameRows

open Nightstream.Implementation.Nebula.ProductionMemoryBatchHashFrameRows

#check prefix_column_values
#check claim_column_values_at
#check batch_column_values
#check input_column_values
#check inputColumns_length

end tests.NebulaProductionMemoryBatchHashFrameRows
