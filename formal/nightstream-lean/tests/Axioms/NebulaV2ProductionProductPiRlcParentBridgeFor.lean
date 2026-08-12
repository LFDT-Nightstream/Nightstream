import Nightstream.Implementation.NebulaV2.ProductionProductPiRlcParentBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionProductPiRlcParentBridgeFor

/-! Dependency gate for the exponent-indexed production PiRLC parent. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductPiRlcParentBridgeFor.parentFields_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductPiRlcParentBridgeFor.parentFields_of_rows

end tests.Axioms.NebulaV2ProductionProductPiRlcParentBridgeFor
