import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe

/-!
Focused elaboration boundary for the activation-compatible exact production
Poseidon2 width-eight permutation occurrence.
-/

namespace NightstreamTests.FPrimeProductionPoseidon2PermutationRecipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe

#check receipt_exact
#check receipt_row_count
#check rows_owned
#check rowIds_nodup
#check initialNumeric_canonical
#check execution_output_eq_semantic
#check gateRow_active_iff
#check gateRows_complete_of_inactive
#check active_sound
#check complete_changesOnly
#check complete_agrees_visible
#check complete_temporary
#check completedNumeric_eq_execution
#check core_complete
#check gateRows_complete_of_active
#check active_complete
#check inactive_complete

end NightstreamTests.FPrimeProductionPoseidon2PermutationRecipe
