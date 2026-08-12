import Nightstream.Implementation.NebulaV2.Production.Memory.BatchPoseidonRows

/-! Regression surface for the exact production memory-batch Poseidon2 rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryBatchPoseidonRows

open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows

#check output_columns_eq_digest
#check trace_rows_length_exact
#check rows_length_exact
#check candidate_row_count_table

end tests.NebulaV2ProductionMemoryBatchPoseidonRows
