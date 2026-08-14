import Nightstream.Implementation.Nebula.Production.Memory.BatchPoseidonRows

/-! Regression surface for the exact production memory-batch Poseidon2 rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryBatchPoseidonRows

open Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonRows

#check output_columns_eq_digest
#check trace_rows_length_exact
#check rows_length_exact
#check candidate_row_count_table

end tests.NebulaProductionMemoryBatchPoseidonRows
