import Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields

/-! Regression surface for the field-native production memory carry. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryCarryFields

open Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields

#check schema_length_exact
#check encode_length
#check tagAt
#check encode_get
#check encode_injective
#check encode_fields_canonical

end tests.NebulaV2ProductionMemoryCarryFields
