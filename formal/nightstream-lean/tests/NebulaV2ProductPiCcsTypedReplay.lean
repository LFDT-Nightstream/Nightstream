import Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay

/-! Regression surface for key-independent typed PiCCS transcript replay. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiCcsTypedReplay

open Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay

#check decodedAlpha_coordinates_eq
#check decodedPoint_coordinates_eq
#check valueReplay_eq_derived_of_serializations
#check valueReplay_eq_derived_of_components

end tests.NebulaV2ProductPiCcsTypedReplay
