import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit

/-!
Focused elaboration boundary for the selected fused production Poseidon2
sponge occurrence on the 23-field plain/stateless XOut preimage.
-/

namespace NightstreamTests.FPrimeProductionPoseidon2Sponge23Recipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open ProductionPoseidon2Sponge23Recipe
open ProductionPoseidon2Sponge23Audit

#check NumericSponge.trace_valid
#check NumericSponge.emissionReceipt
#check receipt_row_count
#check rows_owned
#check rowIds_nodup
#check active_sound
#check complete_changesOnly
#check complete_agrees_visible
#check completedNumeric_eq_execution
#check core_complete
#check active_complete
#check inactive_complete
#check Recipe.rows_supported
#check Recipe.receipt_allocation_count
#check Recipe.receipt_allocation_ids_nodup
#check Recipe.receipt_allocations_owned
#check Recipe.receipt_row_column_conservation
#check Recipe.normalized_row_column_conservation
#check RewriteClass.selected_cost
#check RewriteClass.isolatedCalls_cost
#check RewriteClass.selected_minimum

end NightstreamTests.FPrimeProductionPoseidon2Sponge23Recipe
