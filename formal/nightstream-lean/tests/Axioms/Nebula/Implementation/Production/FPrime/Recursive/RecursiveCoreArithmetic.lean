import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveCoreArithmetic
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionRecursiveCoreArithmetic

/-! Dependency gate for the closed exponent-26 arithmetic certificate. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionRecursiveCoreArithmetic.facts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionRecursiveCoreArithmetic.facts

end tests.Axioms.NebulaProductionRecursiveCoreArithmetic
