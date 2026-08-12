import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.FoldManifestFor
import tests.Axioms.Support

/-! Dependency gate for terminal trailing-fold and close extraction. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionPaperTerminalFoldManifestFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalFoldManifestFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalFoldManifestFor.Program.rows_imply_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_imply_result

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalFoldManifestFor.Result.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Result.exactInvocation

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalFoldManifestFor.Program.satisfies_of_rowsIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.satisfies_of_rowsIncluded

end tests.Axioms.NebulaV2ProductionPaperTerminalFoldManifestFor
