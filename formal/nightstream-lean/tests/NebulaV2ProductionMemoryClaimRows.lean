import Nightstream.Implementation.NebulaV2.ProductionMemoryClaimRows

/-! Regression surface for the production field-native memory-claim decoder. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryClaimRows

open Nightstream.Implementation.NebulaV2.ProductionMemoryClaimRows

#check rows_length_exact
#check rows_imply_exact_claim
#check parsed_unique
#check derive_unique

end tests.NebulaV2ProductionMemoryClaimRows
