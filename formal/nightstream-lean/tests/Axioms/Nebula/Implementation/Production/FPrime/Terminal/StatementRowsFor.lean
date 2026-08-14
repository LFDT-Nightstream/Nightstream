import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.StatementRowsFor
import tests.Axioms.Support

/-! Dependency gate for exact terminal public-statement recomposition rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionPaperTerminalStatementRowsFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalStatementRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalStatementRowsFor.rows_imply_statementPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_statementPlaced

end tests.Axioms.NebulaProductionPaperTerminalStatementRowsFor
