import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductPiDecTypedBridgeFor

/-! Dependency gate for exponent-indexed PiDEC row soundness. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt

end tests.Axioms.NebulaProductPiDecTypedBridgeFor
