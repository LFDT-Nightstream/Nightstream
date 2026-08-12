import Nightstream.Implementation.NebulaV2.Production.Memory.ProductAccumulation
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionMemoryProductAccumulation

open Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation.Run.activeWellFormed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.activeWellFormed

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation.Run.accumulatedProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.accumulatedProductsBalanced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryProductAccumulation.Run.accumulatedFromConcreteOneBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Run.accumulatedFromConcreteOneBalanced

end tests.Axioms.NebulaV2ProductionMemoryProductAccumulation
