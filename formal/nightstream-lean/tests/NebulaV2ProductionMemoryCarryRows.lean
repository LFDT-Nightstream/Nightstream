import Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows

/-! Regression surface for the production field-native memory-carry decoder. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryCarryRows

open Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows

#check rows_length_exact
#check rows_imply_exact_carry
#check parsed_unique
#check derive_unique

end tests.NebulaV2ProductionMemoryCarryRows
