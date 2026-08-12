import Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim

/-! Regression surface for the field-native production full claim. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionFieldNativeFullClaim

open Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim

#check authorityImage_coordinate_count
#check authorityImage_injective_on_canonical
#check nifsInput_eq_recovers_direct_authority_or_collision
#check Value.toProtocolClaim_injective

end tests.NebulaV2ProductionFieldNativeFullClaim
