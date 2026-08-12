import Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreArithmetic
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionRecursiveCoreArithmetic

/-! Dependency gate for the closed exponent-26 arithmetic certificate. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreArithmetic.facts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreArithmetic.facts

end tests.Axioms.NebulaV2ProductionRecursiveCoreArithmetic
