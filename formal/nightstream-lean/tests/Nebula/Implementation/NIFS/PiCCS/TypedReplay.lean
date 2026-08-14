import Nightstream.Implementation.Nebula.NIFS.PiCCS.TypedReplay

/-! Regression surface for key-independent typed PiCCS transcript replay. -/

set_option autoImplicit false

namespace tests.NebulaProductPiCcsTypedReplay

open Nightstream.Implementation.Nebula.ProductPiCcsTypedReplay

#check decodedAlpha_coordinates_eq
#check decodedPoint_coordinates_eq
#check valueReplay_eq_derived_of_serializations
#check valueReplay_eq_derived_of_components

end tests.NebulaProductPiCcsTypedReplay
