import Nightstream.Implementation.Nebula.Production.NIFS.PiRLC.ParentBridgeFor
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionProductPiRlcParentBridgeFor

/-! Dependency gate for the exponent-indexed production PiRLC parent. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionProductPiRlcParentBridgeFor.parentFields_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionProductPiRlcParentBridgeFor.parentFields_of_rows

end tests.Axioms.NebulaProductionProductPiRlcParentBridgeFor
