import Nightstream.Implementation.Nebula.Production.Memory.ClaimRows

/-! Regression surface for the production field-native memory-claim decoder. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryClaimRows

open Nightstream.Implementation.Nebula.ProductionMemoryClaimRows

#check rows_length_exact
#check rows_imply_exact_claim
#check parsed_unique
#check derive_unique

end tests.NebulaProductionMemoryClaimRows
