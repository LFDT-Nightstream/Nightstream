import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.PublicRowsFor
import tests.Axioms.Support

/-! Dependency gate for exact terminal public-result rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionPaperTerminalPublicRowsFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalPublicRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalPublicRowsFor.rows_imply_publicChecks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_publicChecks

end tests.Axioms.NebulaProductionPaperTerminalPublicRowsFor
