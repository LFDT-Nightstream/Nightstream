import Nightstream.Implementation.Nebula.Production.Memory.CarryRows

/-! Regression surface for the production field-native memory-carry decoder. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryCarryRows

open Nightstream.Implementation.Nebula.ProductionMemoryCarryRows

#check rows_length_exact
#check rows_imply_exact_carry
#check parsed_unique
#check derive_unique

end tests.NebulaProductionMemoryCarryRows
