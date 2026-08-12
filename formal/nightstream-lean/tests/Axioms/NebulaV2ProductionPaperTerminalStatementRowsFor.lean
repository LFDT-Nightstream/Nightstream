import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor
import tests.Axioms.Support

/-! Dependency gate for exact terminal public-statement recomposition rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionPaperTerminalStatementRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalStatementRowsFor.rows_imply_statementPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_statementPlaced

end tests.Axioms.NebulaV2ProductionPaperTerminalStatementRowsFor
