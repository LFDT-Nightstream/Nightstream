import Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductPiDecTypedBridgeFor

/-! Dependency gate for exponent-indexed PiDEC row soundness. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt

end tests.Axioms.NebulaV2ProductPiDecTypedBridgeFor
