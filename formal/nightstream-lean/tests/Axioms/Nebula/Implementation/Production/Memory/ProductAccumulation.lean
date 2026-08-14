import Nightstream.Implementation.Nebula.Production.Memory.ProductAccumulation
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionMemoryProductAccumulation

open Nightstream.Implementation.Nebula.ProductionMemoryProductAccumulation

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryProductAccumulation.Run.activeWellFormed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.activeWellFormed

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryProductAccumulation.Run.accumulatedProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.accumulatedProductsBalanced

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryProductAccumulation.Run.accumulatedFromConcreteOneBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.accumulatedFromConcreteOneBalanced

end tests.Axioms.NebulaProductionMemoryProductAccumulation
