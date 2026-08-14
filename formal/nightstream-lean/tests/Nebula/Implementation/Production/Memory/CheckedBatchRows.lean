import Nightstream.Implementation.Nebula.Production.Memory.CheckedBatchRows

/-! Regression surface for production field-native checked memory batches. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryCheckedBatchRows

open Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows

#check Layout.Valid
#check candidate_row_count_table
#check consumesList_of_indexed
#check Result.suffixBatch
#check Result.consumes_suffixBatch
#check derive
#check rows_imply_exact_ordered_batch

end tests.NebulaProductionMemoryCheckedBatchRows
