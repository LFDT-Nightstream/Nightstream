import Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows

/-! Regression surface for production field-native checked memory batches. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryCheckedBatchRows

open Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows

#check Layout.Valid
#check candidate_row_count_table
#check consumesList_of_indexed
#check Result.suffixBatch
#check Result.consumes_suffixBatch
#check derive
#check rows_imply_exact_ordered_batch

end tests.NebulaV2ProductionMemoryCheckedBatchRows
