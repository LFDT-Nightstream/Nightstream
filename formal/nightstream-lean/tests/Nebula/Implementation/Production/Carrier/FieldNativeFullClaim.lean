import Nightstream.Implementation.Nebula.Production.Carrier.FieldNativeFullClaim

/-! Regression surface for the field-native production full claim. -/

set_option autoImplicit false

namespace tests.NebulaProductionFieldNativeFullClaim

open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim

#check authorityImage_coordinate_count
#check authorityImage_injective_on_canonical
#check nifsInput_eq_recovers_direct_authority_or_collision
#check Value.toProtocolClaim_injective

end tests.NebulaProductionFieldNativeFullClaim
