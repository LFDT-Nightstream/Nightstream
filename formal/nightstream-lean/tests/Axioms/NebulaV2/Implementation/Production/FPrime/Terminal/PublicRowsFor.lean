import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.PublicRowsFor
import tests.Axioms.Support

/-! Dependency gate for exact terminal public-result rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionPaperTerminalPublicRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor.rows_imply_publicChecks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_publicChecks

end tests.Axioms.NebulaV2ProductionPaperTerminalPublicRowsFor
